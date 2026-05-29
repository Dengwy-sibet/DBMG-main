import torch
from torch import optim, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
import torchvision.utils as vutils
from tqdm import tqdm
from data.dataset import get_loader
from models.network import Generator, Discriminator
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from utils import denorm, save_checkpoint, load_checkpoint, get_parameter_groups, log_metrics_to_csv
from models.loss import StainLoss, DiceFocalLoss, CONCHPerceptualLoss, LSGANLoss, PyramidLoss, MaskedLSGANLoss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, FrechetInceptionDistance

class Trainer:
    """
    The central optimization framework for training the proposed multi-task Generative Adversarial Network.
    This class handles the Distributed Data Parallel (DDP) execution, mixed-precision optimization,
    piecewise learning rate scheduling, and quantitative/qualitative evaluations.
    """
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.num_workers = config['data']['num_workers']

        # Assign the computation device for the current DDP process
        self.main_device = torch.device(f"cuda:{rank}")

        # Extract optimization hyperparameters
        self.warm_epochs = config['training']['warm_epochs']
        self.decay_epochs = config['training']['decay_epochs']
        self.num_epochs = config['training']['num_epochs']
        self.lr = config['training']['lr']
        self.val_freq = config['training']['val_freq']

        self.img_size = config['training']['img_size']

        # Multi-objective loss weighting coefficients (lambda hyperparameters)
        self.lambda_l1 = config['training']['lambda_l1']          # Weight for spatial reconstruction
        self.lambda_stain = config['training']['lambda_stain']    # Weight for domain-specific stain consistency
        self.lambda_cls = config['training']['lambda_cls']        # Weight for auxiliary classification task
        self.lambda_seg = config['training']['lambda_seg']        # Weight for auxiliary segmentation task
        self.lambda_per = config['training']['lambda_per']        # Weight for deep perceptual feature matching
        self.lambda_gan = config['training']['lambda_gan']        # Weight for adversarial mapping

        self.check_epoch = 0
        self.load_module = config['training']['load_module']
        self.checkpoint_path = config['training']['checkpoint_path']

        # Initialize the Multi-task Generator Network
        self.G = Generator().to(self.main_device)
        # Wrap Generator with DDP. find_unused_parameters=True is enabled to accommodate 
        # multi-task branches (e.g., classification/segmentation) that might dynamically bypass computation.
        self.G = DDP(self.G, device_ids=[rank], find_unused_parameters=True, gradient_as_bucket_view=True)

        # Initialize the Conditional Discriminator Network
        self.D = Discriminator(input_channels=3).to(self.main_device)
        self.D = DDP(self.D, device_ids=[rank], find_unused_parameters=True, gradient_as_bucket_view=True)

        # Retrieve parameter groups with decoupled weight decay for regularization
        self.optim_params_G = get_parameter_groups(self.G, weight_decay=1e-5)
        self.optim_params_D = get_parameter_groups(self.D, weight_decay=1e-5)

        # Initialize AdamW optimizers with specific beta coefficients for stable GAN training
        self.optimizer_G = optim.AdamW(
            self.optim_params_G,
            lr=self.lr,
            betas=(0.5, 0.999),
        )

        self.optimizer_D = optim.AdamW(
            self.optim_params_D,
            lr=self.lr,
            betas=(0.5, 0.999),
        )

        # Resume training from a pre-trained checkpoint if specified
        if self.load_module:
            check_point = load_checkpoint(
                self.checkpoint_path,
                self.G,
                self.D,
                self.optimizer_G, self.optimizer_D,
                is_op=True,
                map_location=self.main_device
            )
            self.check_epoch = check_point['epoch']
            if self.rank == 0:
                print("load checkpoint success")

        # Instantiate objective functions (Criterions)
        self.criterion_gan = LSGANLoss().to(self.main_device)                                   # Least Squares Adversarial Loss
        self.criterion_l1 = PyramidLoss(num_blurs=1, num_downsamples=4).to(self.main_device)    # Multi-scale Structural Loss
        self.criterion_stain = StainLoss(self.main_device)                                      # Histopathological Stain Regularization
        self.criterion_cls = nn.BCEWithLogitsLoss().to(self.main_device)                        # Binary Cross Entropy for Classification
        self.criterion_seg = DiceFocalLoss().to(self.main_device)                               # Hybrid Dice-Focal Loss for Segmentation
        # Foundation Model-based Perceptual Loss (CONCH)
        self.criterion_per = CONCHPerceptualLoss(hf_auth_token="", device=self.main_device)

        self.batch_size = config['training']['batch_size']  

        # Construct distributed data loaders for empirical distribution sampling (Training)
        self.train_loader, self.train_sampler = get_loader(
            root_A=config['data']['train_root_A'],
            root_B=config['data']['train_root_B'],
            mask_root_B=config['data']['train_mask_root_B'],
            img_size=self.img_size,
            batch_size=self.batch_size,
            is_train=True,
            distributed=True,
            rank=rank,
            world_size=self.world_size,
            num_workers=self.num_workers
        )

        # Construct distributed data loaders for inference (Validation)
        self.val_loader, self.val_sampler = get_loader(
            root_A=config['data']['valid_root_A'],
            root_B=config['data']['valid_root_B'],
            mask_root_B=None,
            img_size=self.img_size,
            batch_size=self.batch_size,
            is_train=False,
            distributed=True,
            rank=rank,
            world_size=self.world_size,
            num_workers=self.num_workers
        )
        
        self.save_dir = config['output']['save_dir']
        self.csv_path = config['valid']['csv_path']
        self.save_dir = config['output']['save_dir']
        self.val_output_dir = config['output']['valid_output_dir']

        # Ensure directory existence exclusively on the primary node to avoid race conditions
        if self.rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.val_output_dir, exist_ok=True)

        # Define piecewise learning rate schedule for the Generator:
        # 1. Linear Warm-up strategy to stabilize early training dynamics
        self.scheduler_G1 = LinearLR(
            self.optimizer_G, 
            start_factor=0.1,
            end_factor=1.0, 
            total_iters=self.warm_epochs
        )

        # 2. Constant plateau for feature representation convergence
        self.scheduler_G2 = ConstantLR(
            self.optimizer_G, 
            factor=1.0,
            total_iters=self.decay_epochs
        )

        # 3. Linear decay towards zero for fine-grained optimization
        self.scheduler_G3 = LinearLR(
            self.optimizer_G, 
            start_factor=1.0,
            end_factor=0, 
            total_iters=self.num_epochs-(self.warm_epochs+self.decay_epochs)
        )

        self.scheduler_G = SequentialLR(
            self.optimizer_G, 
            schedulers=[self.scheduler_G1, self.scheduler_G2, self.scheduler_G3],
            milestones=[self.warm_epochs, self.warm_epochs + self.decay_epochs]
        )

        # Mirror the piecewise learning rate schedule for the Discriminator
        self.scheduler_D1 = LinearLR(
            self.optimizer_D, 
            start_factor=0.1,
            end_factor=1.0, 
            total_iters=self.warm_epochs
        )

        self.scheduler_D2 = ConstantLR(
            self.optimizer_D, 
            factor=1.0,
            total_iters=self.decay_epochs
        )

        self.scheduler_D3 = LinearLR(
            self.optimizer_D, 
            start_factor=1.0,
            end_factor=0, 
            total_iters=self.num_epochs-(self.warm_epochs+self.decay_epochs)
        )
        self.scheduler_D = SequentialLR(
            self.optimizer_D, 
            schedulers=[self.scheduler_D1, self.scheduler_D2, self.scheduler_D3],
            milestones=[self.warm_epochs, self.warm_epochs + self.decay_epochs]
        )

        if self.rank == 0:
            print(f"   Mode: DDP (World Size: {world_size})")
            print(f"   Training Batch Size per GPU: {self.batch_size}")

        if self.rank == 0:
            print("Initializing Metrics on GPU...")
            
        # Initialize quantitative evaluation metrics hardware-accelerated on GPU
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.main_device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.main_device)
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(self.main_device)

    def validate(self, current_epoch):
        """
        Executes the validation phase to quantitatively and qualitatively assess the generative performance.
        Computes Fréchet Inception Distance (FID), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity (SSIM).
        """
        # Set the Generator to evaluation mode (disabling Dropout, fixing BatchNorm statistics)
        self.G.eval()

        if self.rank == 0:
            valid_output_dir = os.path.join(self.val_output_dir, f"epoch_{current_epoch}")
            os.makedirs(valid_output_dir, exist_ok=True)

        # Reset metric accumulators prior to iterating over the validation set
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.fid_metric.reset()

        total_val_images = len(self.val_sampler) * self.world_size
        num_to_save = 100

        # Uniformly sample indices for qualitative visualization to limit I/O overhead
        if total_val_images > num_to_save:
            save_indices = set(random.sample(range(total_val_images), num_to_save))
        else:
            save_indices = set(range(total_val_images))

        global_idx = 0

        if self.rank == 0:
            pbar = tqdm(self.val_loader, desc=f"Val Epoch {current_epoch}")
        else:
            pbar = self.val_loader

        # Disable gradient computation for memory-efficient inference
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                real_A = batch['A'].to(self.main_device)
                real_B = batch['B'].to(self.main_device)
                filenames = batch['A_filename']

                # Perform forward pass using Automatic Mixed Precision (AMP)
                with torch.autocast(device_type='cuda'):
                    # The Generator returns a tuple (fake_B, pred_class, pred_mask); extract fake_B for translation
                    fake_B = self.G(real_A)[0]

                # Denormalize tensors from [-1, 1] to [0, 1] for valid metric calculation
                fake_B_norm = denorm(fake_B)
                real_B_norm = denorm(real_B)

                # Accumulate batch statistics for fidelity and perceptual metrics
                self.psnr_metric.update(fake_B_norm, real_B_norm)
                self.ssim_metric.update(fake_B_norm, real_B_norm)
                
                # FID requires computing feature distributions for both true and generated manifolds
                self.fid_metric.update(real_B_norm, real=True)
                self.fid_metric.update(fake_B_norm, real=False)

                batch_size = real_A.size(0)
                for b in range(batch_size):
                    current_image_idx = global_idx + b

                    # Export selected generated samples for qualitative assessment
                    if current_image_idx in save_indices:
                        save_path = os.path.join(valid_output_dir, f"{filenames[b]}.tiff")
                        vutils.save_image(
                            fake_B[b],
                            save_path,
                            normalize=True,
                            value_range=(-1, 1)
                        )

                    global_idx += batch_size

        if self.rank == 0:
            print("Computing metrics...")

        # Aggregate and compute global metric scores across all distributed nodes
        fid_score = self.fid_metric.compute().item()
        psnr_score = self.psnr_metric.compute().item()
        ssim_score = self.ssim_metric.compute().item()

        if self.rank == 0:
            print(
                f"Validation Results - FID: {fid_score:.4f}, PSNR: {psnr_score:.4f}, SSIM: {ssim_score:.4f}")

        return fid_score, psnr_score, ssim_score

    def train(self):
        """
        The main optimization loop formulating the adversarial minimax game:
        min_G max_D V(D, G) along with auxiliary multi-task objectives.
        """
        # Initialize AMP gradient scaler to prevent numerical underflow in float16 gradients
        scaler = torch.amp.GradScaler()

        for epoch in range(self.num_epochs):
            current_epoch = epoch + 1 + self.check_epoch

            # Crucial for DDP: ensure deterministic shuffling sequence across nodes per epoch
            self.train_sampler.set_epoch(current_epoch)

            lr = self.optimizer_G.param_groups[0]['lr']

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0

            if self.rank == 0:
                progress_bar = tqdm(self.train_loader, desc=f"Epoch {current_epoch}/{self.num_epochs} [LR={lr:.6f}]")
            else:
                progress_bar = self.train_loader

            iters = len(self.train_loader)

            # Ensure networks are in training mode (enabling BatchNorm updates, Dropout, etc.)
            self.G.train()
            self.D.train()

            for batch_idx, batch in enumerate(progress_bar):
                # Fetch multi-modal batched data mapping to the corresponding device
                real_A = batch['A'].to(self.main_device)          # Source domain input
                real_B = batch['B'].to(self.main_device)          # Target domain ground truth
                mask_B = batch['B_mask'].to(self.main_device)     # Spatial segmentation mask
                label = batch['label_3plus'].to(self.main_device) # Categorical annotation

                # --- 1. Forward Pass (Generator) ---
                with torch.autocast(device_type='cuda'):
                    # Multi-task generation: Image Synthesis, Classification, and Segmentation
                    fake_B, pred_class, pred_mask = self.G(real_A)

                # --- 2. Optimize Discriminator (D) ---
                self.optimizer_D.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda'):
                    # Evaluate structural realism of the target domain condition on source A
                    pred_real = self.D(real_B, real_A)
                    loss_D_real = self.criterion_gan(pred_real, True, is_disc=True)

                    # Evaluate discriminability of the generated distribution
                    # .detach() is used to prevent gradient propagation back to the Generator
                    pred_fake = self.D(fake_B.detach(), real_A)
                    loss_D_fake = self.criterion_gan(pred_fake, False, is_disc=True)

                    # Aggregate adversarial discriminative loss
                    loss_D = loss_D_real + loss_D_fake

                # Backpropagate scaled gradients and update D parameters
                scaler.scale(loss_D).backward()
                scaler.step(self.optimizer_D)
                loss_D_scalar = loss_D.detach().item()
                epoch_d_loss += loss_D_scalar

                # --- 3. Optimize Generator (G) ---
                self.optimizer_G.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda'):
                    # Calculate generator's ability to deceive the discriminator
                    pred_fake = self.D(fake_B, real_A)
                    
                    # Construct the composite objective function for the Multi-task Generator
                    loss_G_gan = self.criterion_gan(pred_fake, True, is_disc=False) * self.lambda_gan
                    loss_G_l1 = self.criterion_l1(fake_B, real_B) * self.lambda_l1
                    loss_G_stain = self.criterion_stain(fake_B, real_B, mask_B) * self.lambda_stain
                    loss_G_per = self.criterion_per(fake_B, real_B) * self.lambda_per
                    
                    # Auxiliary classification task (an empirical +0.1 offset is applied for robust margin optimization)
                    loss_G_cls = (self.criterion_cls(pred_class, label) + 0.1) * self.lambda_cls 
                    # Auxiliary segmentation task
                    loss_G_seg = self.criterion_seg(pred_mask, mask_B) * self.lambda_seg

                    # Global minimization objective (Total Generator Loss)
                    loss_G = loss_G_gan + loss_G_per + loss_G_stain + loss_G_l1 + loss_G_cls + loss_G_seg

                # Backpropagate scaled gradients and update G parameters
                scaler.scale(loss_G).backward()
                scaler.step(self.optimizer_G)
                scaler.update() # Update the scaling factor for the next iteration
                
                last_g_loss_scalar = loss_G.detach().item()
                epoch_g_loss += last_g_loss_scalar

                # Real-time monitoring of sub-objective convergences
                if self.rank == 0:
                    progress_bar.set_postfix({
                        "cls": f"{loss_G_cls.detach().item():.2f}",
                        "seg": f"{loss_G_seg.detach().item():.2f}",
                        "GAN": f"{loss_G_gan.detach().item():.2f}",
                        "per": f"{loss_G_per.detach().item():.2f}",
                        "l1": f"{loss_G_l1.detach().item():.2f}",
                        "stain": f"{loss_G_stain.detach().item():.2f}",
                        "D_loss": f"{loss_D_scalar:.2f}"
                    })

            # Update learning rates based on the defined polynomial/step policy
            self.scheduler_D.step()
            self.scheduler_G.step()

            avg_g_loss = epoch_g_loss / iters
            avg_d_loss = epoch_d_loss / iters

            # Persist model weights (checkpointing) at epoch granularity
            if self.rank == 0:
                save_checkpoint(self.save_dir, current_epoch,
                                self.G, self.D, self.optimizer_G, self.optimizer_D,
                                f"checkpoint_epoch{current_epoch}---Gloss_{avg_g_loss:.4f}---Dloss_{avg_d_loss:.4f}---lr{lr}.pth.tar")
                print(f"New model at epoch {current_epoch}: {avg_g_loss:.4f},{avg_d_loss:.4f}")

            # Periodically execute evaluation protocol
            if current_epoch % self.val_freq == 0:
                fid, psnr, ssim = self.validate(current_epoch)

                if self.rank == 0:
                    log_metrics_to_csv(self.csv_path, current_epoch, fid, psnr, ssim)

            # Synchronize all processes to prevent desynchronization during subsequent epochs
            dist.barrier()

        # Final serialization of the optimized models
        if self.rank == 0:
            save_checkpoint(self.save_dir, current_epoch,
                            self.G, self.D, self.optimizer_G, self.optimizer_D,
                            "final_module.pth.tar")
            print("Final models saved")