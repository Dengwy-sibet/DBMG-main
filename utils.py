import os.path
import torch
import random
import numpy as np
import torch.nn as nn
import cv2
import tqdm
import torch.distributed as dist
import os
from datetime import datetime
import csv

def setup(rank, world_size):
    """
    Initializes the inter-process communication backend for Distributed Data Parallel (DDP) training.
    Establishes the NCCL (NVIDIA Collective Communications Library) process group 
    to synchronize gradients and buffers across multiple independent GPU processes.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """
    Safely terminates the distributed training environment and destroys the process group 
    to prevent memory leaks and zombie processes.
    """
    dist.destroy_process_group()

def denorm(tensor):
    """
    Reverts the zero-centered normalization [-1, 1] back to the standard image manifold [0, 1].
    Essential for accurate quantitative metric evaluation (e.g., PSNR, SSIM) and visualization.
    """
    return tensor * 0.5 + 0.5

def set_seed(seed):
    """
    Enforces strict experimental reproducibility by fixing the Pseudo-Random Number Generator (PRNG) 
    states across all hardware and software backends (Python, NumPy, PyTorch, and cuDNN).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force cuDNN to select deterministic convolution algorithms
        torch.backends.cudnn.deterministic = True
        # Disable heuristic algorithm benchmarking to ensure execution graph consistency
        torch.backends.cudnn.benchmark = False

def get_parameter_groups(model, weight_decay, skip_list=()):
    """
    Orthogonal Weight Decay Regularization Strategy.
    Decouples weight decay application from normalization layers (e.g., LayerNorm/BatchNorm weights) 
    and bias terms. Applying weight decay to these 1D structural parameters can severely 
    degrade the expressiveness of the network and lead to suboptimal convergence.
    """
    decay = []
    no_decay =[]
    for name, param in model.named_parameters():
            
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
            
    return[
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ]

def save_checkpoint(save_dir, epoch, G, D, OP_G, OP_D, filename):
    """
    Serializes and persists the complete internal state of the optimization framework.
    Captures multi-stage parameters including the Generator, Discriminator, and 
    their respective optimizer momentum buffers to allow seamless training resumption.
    """
    state = {
        'epoch': epoch,
        'G': G.state_dict(),
        'D': D.state_dict(),
        'OP_G_state_dict': OP_G.state_dict(),
        'OP_D_state_dict': OP_D.state_dict(),
    }
    torch.save(state, os.path.join(save_dir, filename))
    return state

def load_checkpoint(checkpoint_file, G, D, OP_G, OP_D, is_op):
    """
    Deserializes and maps pre-trained architectural parameters and optimization states 
    back into the initialized computational graph.
    """
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
    G.load_state_dict(checkpoint['G'])
    D.load_state_dict(checkpoint['D'])
    if is_op:
        OP_G.load_state_dict(checkpoint['OP_G_state_dict'])
        OP_D.load_state_dict(checkpoint['OP_D_state_dict'])
    return checkpoint

def load_savemodel(save_path, base_model, device_ids):
    """
    Specialized deserialization pipeline for high-throughput inference mapping.
    Resolves weight dict key inconsistencies and wraps the underlying network within 
    a DataParallel (DP) context for scalable multi-GPU evaluation.
    """
    device_ids = device_ids
    main_device = torch.device(f"cuda:{device_ids[0]}")

    base_model = base_model
    checkpoint = torch.load(save_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'G' in checkpoint:
        state_dict = checkpoint['G']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model = nn.DataParallel(base_model, device_ids=device_ids).to(main_device)
    model.load_state_dict(state_dict, strict=True)

    return model

def tensor_to_bgr(tensor):
    """
    Transforms continuous tensor representations [0, 1] into discrete 8-bit unsigned integer 
    matrices and converts the color space from RGB to BGR for standard OpenCV I/O operations.
    """
    img_np = np.clip(tensor.cpu().permute(1, 2, 0).numpy(), 0, 1)
    img_uint8 = (img_np * 255).astype(np.uint8)
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

def generate_and_save(current_epoch, test_loader, test_output_dir, model, device_ids):
    """
    Executes the deterministic evaluation protocol.
    Iteratively translates source domain instances into the target semantic space utilizing 
    Automatic Mixed Precision (AMP), and archives the synthesized high-fidelity outputs.
    """
    # Fixate batch normalization moving statistics and disable dropout stochasticity
    model.eval()
    test_dir = os.path.join(test_output_dir, f"epoch_{current_epoch}")
    os.makedirs(test_dir, exist_ok=True)
    main_device = torch.device(f"cuda:{device_ids[0]}")

    # Suspend computational graph history tracking to significantly reduce VRAM footprint
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Generating Epoch {current_epoch}")):
            real_A = batch['A'].to(main_device)
            filenames = batch['A_filename']

            # Hardware-accelerated inference pass utilizing FP16/BF16 tensor cores
            with torch.autocast(device_type='cuda'):
                fake_B = model(real_A)

            # Spatial mapping and post-processing for disk archival
            fake_B_norm = denorm(fake_B)
            img_fake_bgr = tensor_to_bgr(fake_B_norm[0])

            save_path = os.path.join(test_dir, f"{filenames[0]}.tiff")
            cv2.imwrite(save_path, img_fake_bgr)

def log_metrics_to_csv(csv_filepath, epoch, fid, ssim, psnr):
    """
    Quantitative Evaluation Logging Utility.
    Persistently records the trajectory of generative performance metrics across training epochs.
    Tracks core statistical and perceptual metrics (FID, SSIM, PSNR) to facilitate 
    downstream empirical analysis, plotting, and convergence monitoring.
    """
    # Define the structural schema for the evaluation log. 
    fieldnames = ['epoch', 'timestamp', 'fid', 'ssim', 'psnr']
    
    # Probe the existence of the archival file to conditionally initialize the structural header
    file_exists = os.path.exists(csv_filepath)
    try:
        # Utilize append mode ('a') to continuously track metrics without overwriting previous historical states
        with open(csv_filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            # Construct the metric payload.
            # Enforces a standardized floating-point precision (4 decimal places) for rigorous quantitative reporting,
            # while gracefully falling back to 'NaN' (Not a Number) to handle missing measurements during partial evaluations.
            row = {
                'epoch': epoch,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'fid': f"{fid:.4f}" if fid is not None else 'NaN',
                'ssim': f"{ssim:.4f}" if ssim is not None else 'NaN',
                'psnr': f"{psnr:.4f}" if psnr is not None else 'NaN'
            }
            writer.writerow(row)
            
    except Exception as e:
        # Non-blocking exception handling to ensure that minor disk I/O bottlenecks or permission faults 
        # do not fatally interrupt the computationally expensive training pipeline.
        print(f"Error logging: {e}")
