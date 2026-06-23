import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.stats import pearsonr

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
    KernelInceptionDistance
)

# Allocate computations to hardware accelerators (GPU) if available for high-throughput metric evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-calibrated color deconvolution coefficients.
# These specific vectors project the Optical Density (OD) space representations into 
# the isolated DAB (3,3'-Diaminobenzidine, typically brown) stain concentration channel.
C0 = torch.tensor(-1.00767869, device=device)
C1 = torch.tensor(1.13473037, device=device)
C2 = torch.tensor(-0.48041419, device=device)

def get_image_files_without_ext(directory, is_fake=False):
    """
    Utility function to robustly pair synthesized (fake) representations with their 
    corresponding ground-truth (GT) references by resolving heterogeneous file naming 
    conventions and automatically stripping algorithmic suffixes.
    """
    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    gan_suffixes =['_fake_B', '_pre_B', '_synthesized', '_fake', '_out']
    
    files = {}
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if ext.lower() in supported_ext:
            if is_fake:
                # Strip known generator suffixes to align the key with the GT filename
                for suffix in gan_suffixes:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                        break
            files[name] = filename
    return files

def load_image_as_tensor(path):
    """
    Standardized image loading pipeline. Decodes the image, maps intensities 
    to the [0, 1] continuous manifold, and prepares the batch dimension for GPU processing.
    """
    with Image.open(path) as img:
        img_rgb = img.convert('RGB')
        tensor = torch.from_numpy(np.array(img_rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device)

@torch.no_grad()
def extract_clinical_features_from_tensors(img_gt, img_fake, patch_size=256, od_threshold=0.1):
    """
    Domain-Specific Clinical Metric Extraction Framework.
    Mimics digital pathology quantification algorithms by extracting Integrated Optical Density (IOD) 
    and localized positive staining areas, evaluating the biological reliability of the synthesized images.
    """
    # Formulate a batched tensor to accelerate parallel operations on both GT and fake images
    batch_rgb = torch.cat([img_gt, img_fake], dim=0)

    # Segment the biologically relevant tissue Region of Interest (ROI).
    # Filters out high-luminance background components (e.g., glass slide) and extreme artifacts.
    gray = batch_rgb.mean(dim=1) 
    tissue_mask = (gray < 0.86) & (gray > 0.05)

    # Beer-Lambert Law application: Map normalized RGB intensities to Optical Density (OD) space.
    # A small epsilon (1e-6) is utilized to ensure numerical stability during the logarithmic transformation.
    batch_rgb = torch.clamp(batch_rgb, 1e-6, 1.0)
    OD = -torch.log(batch_rgb)

    # Execute linear color deconvolution to isolate the target IHC biomarker (e.g., HER2 DAB stain).
    dab_od = OD[:, 0, :, :] * C0 + OD[:, 1, :, :] * C1 + OD[:, 2, :, :] * C2
    dab_od = dab_od * tissue_mask
    dab_od = torch.clamp(dab_od, min=0.0)

    # Quantify the total DAB mass equivalent (Integrated Optical Density - IOD) across the valid tissue mask.
    total_iods = dab_od.sum(dim=(1, 2)).cpu().numpy()
    iod_gt, iod_fake = float(total_iods[0]), float(total_iods[1])

    # Binarize the continuous DAB concentration utilizing a biologically motivated threshold.
    positive_mask = (dab_od > od_threshold).float()
    B, H, W = positive_mask.shape

    # Dynamically pad spatial dimensions to ensure perfect divisibility by the specified patch size.
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        positive_mask = F.pad(positive_mask, (0, pad_w, 0, pad_h), mode='constant', value=0)

    # Extract non-overlapping spatial patches for localized morphological evaluation.
    patches = positive_mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patch_areas = patches.sum(dim=(3, 4))
    
    # Flatten the localized statistical distributions for downstream correlation analyses.
    areas_gt = patch_areas[0].flatten().cpu().numpy().tolist()
    areas_fake = patch_areas[1].flatten().cpu().numpy().tolist()

    return iod_gt, iod_fake, areas_gt, areas_fake

def evaluate_all_metrics(gt_dir, model_dirs_dict):
    """
    Orchestrates the holistic evaluation protocol.
    Iteratively assesses multi-dimensional generative performance spanning:
    1. Pixel-level fidelity (PSNR)
    2. Structural coherence (SSIM, MS-SSIM)
    3. Deep perceptual similarity (LPIPS)
    4. Distributional discrepancy (FID, KID)
    5. Clinical pathology quantification (IOD Pearson Correlation & Mean Absolute Offset)
    """
    # Initialize hardware-accelerated evaluators for canonical generative metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid_metric = KernelInceptionDistance(subset_size=500, subsets=20, normalize=True).to(device)
    
    results_list =[]
    gt_images_dict = get_image_files_without_ext(gt_dir, is_fake=False)
    
    for model_name, fake_dir in model_dirs_dict.items():
        print(f"start: {model_name}")
        fake_images_dict = get_image_files_without_ext(fake_dir, is_fake=True)
        
        # Reset distributional accumulators prior to iterating over a new generative model
        fid_metric.reset()
        kid_metric.reset()
        total_psnr, total_ssim, total_msssim, total_lpips = 0.0, 0.0, 0.0, 0.0
        
        all_gt_iods, all_fake_iods = [],[]
        all_gt_patch_areas, all_fake_patch_areas = [], []
        abs_offsets = []
        
        valid_count = 0
        pbar = tqdm(gt_images_dict.keys(), desc="Processing")
        
        for img_base_name in pbar:
            if img_base_name not in fake_images_dict:
                continue
            
            gt_path = os.path.join(gt_dir, gt_images_dict[img_base_name])
            fake_path = os.path.join(fake_dir, fake_images_dict[img_base_name])
            
            try:
                img_gt = load_image_as_tensor(gt_path)
                img_fake = load_image_as_tensor(fake_path)

                # Accumulate image-level deterministic metrics
                total_psnr += psnr_metric(img_fake, img_gt).item()
                total_ssim += ssim_metric(img_fake, img_gt).item()
                total_msssim += msssim_metric(img_fake, img_gt).item()
                total_lpips += lpips_metric(img_fake, img_gt).item()

                # Accumulate deep feature mappings for dataset-level distributional evaluations (FID/KID)
                fid_metric.update(img_gt, real=True)
                fid_metric.update(img_fake, real=False)
                kid_metric.update(img_gt, real=True)
                kid_metric.update(img_fake, real=False)

                # Extract domain-specific biological quantification signals
                iod_gt, iod_fake, areas_gt, areas_fake = extract_clinical_features_from_tensors(img_gt, img_fake)
                
            except Exception as e:
                # Non-blocking error handling to ensure uninterrupted large-scale dataset evaluation
                tqdm.write(f"\n pass image {img_base_name} error: {e}")
                continue

            all_gt_iods.append(iod_gt)
            all_fake_iods.append(iod_fake)
            all_gt_patch_areas.extend(areas_gt)
            all_fake_patch_areas.extend(areas_fake)
            
            # Record the absolute quantification error for systematic bias analysis
            offset = iod_fake - iod_gt
            abs_offsets.append(abs(offset))
            
            valid_count += 1
            # Periodically invoke the Python garbage collector to prevent VRAM/RAM out-of-memory 
            # anomalies during extensive multi-model dataset evaluations.
            if valid_count % 30 == 0:
                gc.collect() 
                
        if valid_count == 0:
            print(f"error：model {model_name} can't find image")
            continue

        print(f"Computing FID/KID and Pearson statistics....")
        
        # Aggregate and average the quantitative records
        avg_psnr = total_psnr / valid_count
        avg_ssim = total_ssim / valid_count
        avg_msssim = total_msssim / valid_count
        avg_lpips = total_lpips / valid_count
        fid_score = fid_metric.compute().item()
        kid_score = kid_metric.compute()[0].item()

        # Evaluate the linear correlation of the clinical biomarker expression between synthesized and true domains.
        # Fallback to 0.0 to handle mathematically degenerate cases (e.g., zero variance).
        if np.std(all_gt_iods) > 0 and np.std(all_fake_iods) > 0:
            iod_pearson = pearsonr(all_gt_iods, all_fake_iods)[0]
        else:
            iod_pearson = 0.0
                      
        mean_abs_offset = np.mean(abs_offsets)
        
        # Construct the final comprehensive performance profile
        results_list.append({
            "Model": model_name,
            "PSNR (↑)": round(avg_psnr, 4),
            "SSIM (↑)": round(avg_ssim, 4),
            "MS-SSIM (↑)": round(avg_msssim, 4),
            "LPIPS (↓)": round(avg_lpips, 4),
            "FID (↓)": round(fid_score, 4),
            "KID (↓)": round(kid_score, 4),
            "IOD Pearson-R (↑)": round(iod_pearson, 4),
            "IOD Mean Abs Offset (↓)": round(mean_abs_offset, 4),
        })

    # Persist the experimental results into a structured format for statistical reporting
    df_results = pd.DataFrame(results_list)
    print("\n" + "="*110)
    print("Summary of final comprehensive evaluation results")
    print("="*110)
    print(df_results.to_string(index=False))
    
    csv_name = "comprehensive_evaluation_results.csv"
    df_results.to_csv(csv_name, index=False)
    print("="*110)
    print(f"All data have been saved to: {csv_name}")

if __name__ == "__main__":

    #Path to the folder containing real IHC images
    GROUND_TRUTH_DIR = ""  
    
    #Path to the folder containing generated IHC images from each model
    MODELS_TO_TEST = {
        #"CycleGAN": "",
        #"Pyramidp2p": "",
        #"ASP": "",
        #"CSSP2P": "",
        #"PSPStain": "",
        "DBMG": "",
    }
    
    evaluate_all_metrics(GROUND_TRUTH_DIR, MODELS_TO_TEST)