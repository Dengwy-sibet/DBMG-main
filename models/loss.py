import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import random
from torchvision import transforms
from conch.open_clip_custom import create_model_from_pretrained

class PyramidLoss(nn.Module):
    """
    Multi-Scale Pyramid Loss.
    Enforces structural and low-frequency consistency between the generated and target images
    across varying spatial resolutions via a Gaussian pyramid decomposition.
    """
    def __init__(self, num_blurs, num_downsamples):
        super(PyramidLoss, self).__init__()

        self.kernel_size = (3, 3)     
        self.sigma = (1.0, 1.0)        
        self.pool_stride = 2           
        self.pool_kernel = 1           
        
        self.num_blurs = num_blurs     
        self.num_downsamples = num_downsamples 
        
        # We employ Mean Absolute Error (L1) at each pyramid level to preserve high-frequency details 
        # better than L2 penalty, reducing over-smoothing.
        self.l1 = nn.L1Loss(reduction='mean')

    def _process_image(self, img):
        """
        Applies a low-pass Gaussian filter followed by spatial downsampling 
        to extract hierarchical representations.
        """
        input = img
        for _ in range(self.num_blurs):
            input = kornia.filters.gaussian_blur2d(
            input, 
            kernel_size=self.kernel_size, 
            sigma=self.sigma
            )
        
        output = kornia.filters.blur_pool2d(
            input, 
            kernel_size=self.pool_kernel, 
            stride=self.pool_stride
        )

        return output

    def forward(self, fake_B, real_B):
        # Accumulate pixel-wise discrepancies across multiple progressive downsampling stages
        total_loss = 0.0
        fake_B_current = fake_B
        real_B_current = real_B
        
        for _ in range(1, self.num_downsamples + 1):
            fake_B_current = self._process_image(fake_B_current)
            real_B_current = self._process_image(real_B_current)

            level_loss = self.l1(fake_B_current, real_B_current) 
            total_loss += level_loss
        
        return total_loss
    
class LSGANLoss(nn.Module):
    """
    Least Squares Generative Adversarial Network (LSGAN) Loss.
    Replaces the standard cross-entropy objective with a least squares formulation 
    to mitigate vanishing gradients and stabilize the min-max adversarial game.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() 

    def forward(self, pred, target_is_real, is_disc=True):
        if is_disc:
            # Discriminator objective: push real samples to 1, generated samples to 0
            if target_is_real:
                loss = self.mse(pred, torch.ones_like(pred))
            else:
                loss = self.mse(pred, torch.zeros_like(pred))
        else:
            # Generator objective: fool the discriminator by pushing generated samples towards 1
            loss = self.mse(pred, torch.ones_like(pred))
        return loss

class MaskedLSGANLoss(nn.Module):
    """
    Spatially-Aware Masked LSGAN Loss.
    Combines a global image-level adversarial objective with a localized, region-of-interest (ROI) 
    adversarial penalty to enhance synthesis fidelity in highly specific histological regions.
    """
    def __init__(self, global_weight=1.0, regional_weight=2.0):
        super().__init__()
        self.mse = nn.MSELoss() 
        self.global_weight = global_weight
        self.regional_weight = regional_weight

    def forward(self, preds_dict, target_is_real, is_disc=True):

        global_pred = preds_dict["global_pred"]
        regional_pred = preds_dict["regional_pred"]
        mask = preds_dict["mask_downsampled"]

        if is_disc:
            target_val = 1.0 if target_is_real else 0.0
        else:
            target_val = 1.0

        # Global adversarial mapping
        target_global = torch.full_like(global_pred, target_val)
        loss_global = self.mse(global_pred, target_global)

        # Region-specific adversarial mapping focused exclusively on masked semantic areas
        target_regional = torch.full_like(regional_pred, target_val)
        squared_diff = (regional_pred - target_regional) ** 2
        loss_regional = (squared_diff * mask).sum() / (mask.sum() + 1e-8)

        # Aggregate holistic and localized losses
        total_loss = (self.global_weight * loss_global) + (self.regional_weight * loss_regional)
        
        return total_loss
    
    
class StainLoss(nn.Module):
    """
    Domain-Specific Histopathological Stain Loss.
    Leverages the Beer-Lambert law to map RGB color space into Optical Density (OD) space.
    Performs color deconvolution to isolate specific stain concentrations (e.g., DAB) 
    and enforces statistical matching (histogram and total mass) between source and generated domains.
    """
    def __init__(self, device):
        super(StainLoss, self).__init__()
        self.device = device
        self.num_bins = 20
        self.mse = nn.MSELoss(reduction='none')

        # Empirical stain unmixing matrix (Color Deconvolution) for histopathological domains
        stain_matrix = torch.tensor([[0.6500286, 0.704031, 0.2860126],  
                                    [0.2681475, 0.570313, 0.7764271],  
                                    [0.71102734, 0.42318153, 0.5615672]
        ], device=device, dtype=torch.float32)

        # L2 normalize the stain matrix along the channel dimension
        row_norms = torch.norm(stain_matrix, dim=1, keepdim=True)
        stain_matrix = stain_matrix / row_norms

        # Compute the projection operator (inverse or pseudo-inverse) for OD to Stain transformation
        try:
            self.Q = torch.linalg.inv(stain_matrix)
        except:
            self.Q = torch.linalg.pinv(stain_matrix)

    def calculate_histo_sums(self, features, num_histos, min_val, max_val):
        """
        Approximates a differentiable histogram calculation via scatter addition 
        to evaluate the distribution discrepancy of stain intensities.
        """
        B, N = features.shape
        bucket_width = (max_val - min_val) / num_histos

        # Quantize feature intensities into discrete bin indices
        histo_indices = ((features - min_val) / bucket_width).clamp(0, num_histos - 1).long()
        
        batch_sums = torch.zeros((B, num_histos), device=features.device, dtype=features.dtype)
        batch_sums.scatter_add_(1, histo_indices, features)
        
        return batch_sums
    
    def get_fiji_style_channels(self, img_tensor):
        """
        Transforms standard RGB tensors into decoupled stain concentrations mimicking 
        the Fiji/ImageJ Color Deconvolution algorithm.
        """
        B, C, H, W = img_tensor.shape
        # Re-scale from [-1, 1] normalized space to standard [0, 255] RGB intensity space
        img_float = (img_tensor + 1.0) * 127.5
        
        # Dynamically estimate the brightest reference point (background illumination)
        flat_img = img_float.view(B, C, -1)
        brightest_ref = torch.quantile(flat_img, 0.99, dim=2).view(B, C, 1, 1)

        numerator = img_float + 1.0
        denominator = brightest_ref + 1.0

        # Beer-Lambert mapping to Optical Density (OD) Space
        epsilon = 1e-7
        ratio = numerator / (denominator + epsilon)
        od = -torch.log10(torch.clamp(ratio, min=epsilon))
        od = torch.relu(od)

        # Project OD representations into orthogonal stain components
        od_permuted = od.permute(0, 2, 3, 1)
        concentrations = torch.matmul(od_permuted, self.Q)

        # Extract the target stain channel (e.g., DAB / brown marker)
        dab_conc = concentrations[..., 1]

        def post_process_channel(conc):
            # Normalize concentration values robustly utilizing quantile clipping
            flat_conc = conc.view(B, -1)
            max_val = torch.quantile(flat_conc, 0.999, dim=1, keepdim=True).view(B, 1, 1)
            max_val = torch.clamp(max_val, min=epsilon)
            norm = torch.clamp(conc / max_val, 0.0, 1.0)
            return norm

        dab_final = post_process_channel(dab_conc) 
        return dab_final.unsqueeze(1)

    def forward(self, fake_B, real_B, mask_B):
        # Disable automatic mixed precision for numerical stability during log10 calculations
        with torch.autocast(enabled=False, device_type='cuda'):
            fake_B = fake_B.float()
            real_B = real_B.float()
            
            # Decouple the target stain and isolate the foreground via semantic mask
            fake_dab = self.get_fiji_style_channels(fake_B)
            fake_dab_clean = fake_dab * (mask_B > 0.0).float()

            with torch.no_grad():
                real_dab = self.get_fiji_style_channels(real_B)
                real_dab_clean = real_dab * (mask_B > 0.0).float()

            B, C, H, W = fake_dab.shape

            # Compute macro-level discrepancy (Total staining mass equivalent)
            fake_total_sum = fake_dab_clean.sum(dim=(1, 2, 3))
            real_total_sum = real_dab_clean.sum(dim=(1, 2, 3))

            loss_avg = self.mse(fake_total_sum, real_total_sum) / (H * W) ** 2

            # Compute micro-level discrepancy (Staining intensity distribution)
            fake_flat = fake_dab_clean.view(B, -1)
            real_flat = real_dab_clean.view(B, -1)
            
            fake_histo = self.calculate_histo_sums(fake_flat, self.num_bins, 0.0, 1.0)
            real_histo = self.calculate_histo_sums(real_flat, self.num_bins, 0.0, 1.0)
            
            loss_histo = (((fake_histo / (H * W) - real_histo / (H * W))**2).sum(1))

            # Adaptive fusion: If the total mass ratio is within a reliable margin [0.6, 1.4], 
            # exclusively regularize the distribution (histogram). Otherwise, regularize both.
            ratio = fake_total_sum / (real_total_sum + 1e-8)
            condition = (ratio >= 0.6) & (ratio <= 1.4)
            loss_dab = torch.where(condition, loss_histo, loss_avg + loss_histo).mean()

            return loss_dab

class DiceFocalLoss(nn.Module):
    """
    Hybrid Segmentation Objective.
    Focal Loss provides dense pixel-wise supervision and dynamically scales gradients 
    based on prediction confidence to handle extreme foreground-background class imbalance.
    Dice Loss evaluates the regional overlap (Intersection-over-Union mapping), 
    providing robust performance independent of ROI size.
    """
    def __init__(self, gamma=2.0, alpha=0.9, lambda_dice=1.0, lambda_focal=2.0, smooth=1.0, reduction='mean'):
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        # Construct spatially-weighted Focal formulation
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = focal_weight * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        pred_sigmoid = torch.sigmoid(pred)
        
        if self.reduction == 'none':
            pass
        else:
            # Flatten spatial dimensions for global set-theoretic evaluation
            pred_flat = pred_sigmoid.view(-1)
            target_flat = target.view(-1)
            
            target_sum = target_flat.sum()
            
            if target_sum == 0:
                dice_loss = torch.tensor(0.0, device=pred.device)
            else:
                intersection = (pred_flat * target_flat).sum()
                union = pred_flat.sum() + target_sum
                dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)

        # Composite minimization criterion
        total_loss = self.lambda_focal * focal_loss + self.lambda_dice * dice_loss
        
        return total_loss
            
class CONCHPerceptualLoss(nn.Module):
    """
    Foundation Model-driven Perceptual Fidelity Loss.
    Utilizes semantic embeddings from 'CONCH' (a vision-language foundational model 
    specialized in computational pathology) to quantify the high-level semantic 
    and spatial representation divergence between real and generated images.
    """
    def __init__(self, hf_auth_token, device, crop_size=256):
        super().__init__()
        self.device = device
        self.crop_size = crop_size

        # Instantiate pre-trained Vision Transformer (ViT) encoder from CONCH
        model, _ = create_model_from_pretrained(
            'conch_ViT-B-16', 
            "hf_hub:MahmoodLab/conch", 
            hf_auth_token=hf_auth_token
        )
        self.vision_encoder = model.visual.to(device)
        self.vision_encoder.eval() 

        # Freeze backpropagated gradients for the foundation model to serve strictly as an evaluator
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Intercept hierarchical intermediate representations from selected transformer blocks
        self.target_blocks =[3, 6, 9, 11]
        self.features =[]
        
        def hook_fn(module, input, output):
            self.features.append(output)

        for idx in self.target_blocks:
            self.vision_encoder.trunk.blocks[idx].register_forward_hook(hook_fn)

        # Standard statistical normalization matching the pre-training protocol of the foundation model
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        self.l1_loss = nn.L1Loss()

    def extract_features(self, x):
        """
        Executes inference to retrieve the hooked intermediate embeddings.
        """
        self.features.clear() 
        with torch.no_grad(): 
            _ = self.vision_encoder(x)
        return list(self.features)

    def forward(self, fake_B, real_B):  
        B, C, H, W = fake_B.shape

        # Stochastic patch-cropping to alleviate GPU memory overhead 
        # while preserving fine-grained localized structural evaluation.
        if H > self.crop_size or W > self.crop_size:
            top = random.randint(0, H - self.crop_size)
            left = random.randint(0, W - self.crop_size)
            
            fake_crop = fake_B[:, :, top:top+self.crop_size, left:left+self.crop_size]
            real_crop = real_B[:, :, top:top+self.crop_size, left:left+self.crop_size]
        else:
            fake_crop, real_crop = fake_B, real_B

        # Map [-1, 1] normalized synthesis outputs into standard pre-trained distributions
        fake_crop = self.normalize((fake_crop + 1.0) / 2.0)
        real_crop = self.normalize((real_crop + 1.0) / 2.0)

        feat_fake = self.extract_features(fake_crop)
        feat_real = self.extract_features(real_crop)

        total_loss = 0.0

        for f_fake, f_real in zip(feat_fake, feat_real):
            # 1. Semantic Token alignment: evaluates global semantic congruence via the ViT CLS token
            cls_fake = f_fake[:, 0, :]   
            cls_real = f_real[:, 0, :]
            loss_cls = (1.0 - F.cosine_similarity(cls_fake, cls_real, dim=-1)).mean()

            # 2. Spatial Token alignment: evaluates patch-wise structural consistency via sequence tokens
            spatial_fake = f_fake[:, 1:, :] 
            spatial_real = f_real[:, 1:, :]
            loss_spatial = (1.0 - F.cosine_similarity(spatial_fake, spatial_real, dim=-1)).mean()

            total_loss += (loss_cls + loss_spatial)

        # Average contextual divergence across varying levels of abstraction
        return total_loss / len(self.target_blocks)