import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from torch.nn.utils import spectral_norm

class LayerNorm2d(nn.Module):
    """
    Channel-wise Layer Normalization adapted for 2D spatial feature maps.
    Computes normalization across the channel dimension while maintaining the spatial resolution,
    often utilized as a standard normalization scheme in Vision Transformers and ConvNeXt architectures.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.normalized_shape = (num_channels,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute from NCHW to NHWC for efficient LayerNorm computation, then revert to NCHW
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
        ).permute(0, 3, 1, 2)


class GRN(nn.Module):
    """ 
    Global Response Normalization (GRN) layer introduced in ConvNeXt V2.
    Enhances inter-channel feature competition and spatial representation quality 
    by normalizing the aggregated global spatial responses of each channel.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # Compute the L2 norm of spatial features for each channel
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # Normalize responses relative to the mean spatial response across all channels
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlockV2(nn.Module):
    """
    Core building block of the ConvNeXt V2 architecture.
    Employs an inverted bottleneck design with depthwise convolutions, Global Response 
    Normalization (GRN), and stochastic depth (DropPath) for regularized representation learning.
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # Large-kernel (7x7) depthwise convolution to capture expansive receptive fields
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = LayerNorm2d(dim)
        # Pointwise convolutions projecting features to a higher-dimensional space (expansion ratio = 4)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim) 
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # Regularization via DropPath (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Learnable scaling parameter for residual connections initialized with a small value
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # Format transformation: NCHW -> NHWC for linear layers
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x) 
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # Revert format: NHWC -> NCHW
        x = input + self.drop_path(x)
        return x

class DecoderResBlock(nn.Module):
    """
    Residual Refinement Block utilized in the decoding pathway.
    Extracts and reconstructs high-frequency spatial details using sequential convolutions, 
    Layer Normalization, and GELU activations to prevent feature degradation during upsampling.
    """
    def __init__(self, dim):
        super().__init__()
        # Reflect padding is adopted to mitigate boundary artifacts during synthesis
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.norm1 = LayerNorm2d(dim)
        self.act = nn.GELU()

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.norm2 = LayerNorm2d(dim)
        # Zero-initialized residual scaling allows the block to act as an identity mapping initially
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        shortcut = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return shortcut + self.gamma * x
    

class ConvNeXtStage(nn.Module):
    """
    Hierarchical stage of the ConvNeXt encoder.
    Consists of an optional spatial downsampling module followed by a sequence of ConvNeXt V2 blocks.
    """
    def __init__(self, in_dim, out_dim, stride=1, depth=2, drop_path=0.):
        super().__init__()
        # Spatially downsample and transition channel dimensions via a 2x2 convolution with stride 2
        if stride > 1 or in_dim != out_dim:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_dim),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=stride, bias=False)
            )
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(*[
            ConvNeXtBlockV2(dim=out_dim, drop_path=drop_path)
            for _ in range(depth)
        ])

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXtEncoder(nn.Module):
    """
    Multi-scale hierarchical feature extractor serving as the backbone.
    Progressively extracts abstract semantic representations at four varying spatial resolutions.
    """
    def __init__(self, in_chans=3, embed_dim=96, depths=[3, 3, 9, 3]):
        super().__init__()

        # Stem layer: Initial heavy downsampling (stride=4 overall) for input resolution reduction
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='reflect'),
            LayerNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False, padding_mode='reflect'),
            LayerNorm2d(embed_dim)
        )
        self.stages = nn.ModuleList()

        # Isotropic dimension expansion defining the hierarchical pyramid
        dims =[embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        self.stages.append(ConvNeXtStage(embed_dim, dims[0], stride=1, depth=depths[0]))
        self.stages.append(ConvNeXtStage(dims[0], dims[1], stride=2, depth=depths[1]))
        self.stages.append(ConvNeXtStage(dims[1], dims[2], stride=2, depth=depths[2]))
        self.stages.append(ConvNeXtStage(dims[2], dims[3], stride=2, depth=depths[3]))

    def forward(self, x):
        features =[]
        x = self.stem(x)
        # Retain intermediate multi-scale representations for downstream dense prediction tasks
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features 



class SkipRefine(nn.Module):
    """
    Skip Connection Refinement Module.
    Integrates dual-attention mechanisms (Channel-wise Squeeze-and-Excitation and Spatial Gating) 
    to selectively emphasize semantically relevant features and suppress background noise 
    before bridging encoder features to the decoder.
    """
    def __init__(self, dim):
        super().__init__()
        # Channel Attention Component (Squeeze-and-Excitation)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial Attention Component
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Element-wise feature modulation using both channel-wise and spatial-wise attention maps
        return x * self.se(x) * self.spatial_gate(x)


class CNNDecoderStage(nn.Module):
    """
    Progressive upsampling and feature fusion stage in the generator's decoding pathway.
    Concatenates upsampled deep features with structurally rich shallower features via skip connections.
    """
    def __init__(self, in_dim, skip_dim, out_dim, num_blocks=2):
        super().__init__()
        # Spatial resolution doubling via bilinear interpolation
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, padding_mode='reflect', bias=False), 
            LayerNorm2d(out_dim),
            nn.GELU()
        )

        # Cross-level semantic fusion mapping
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim + skip_dim, out_dim, 1, bias=False),
            LayerNorm2d(out_dim),
            nn.GELU()
        )

        # Deep refinement utilizing cascaded residual blocks
        self.blocks = nn.Sequential(*[
            DecoderResBlock(dim=out_dim) for _ in range(num_blocks)
        ])

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1) # Dense skip connection
        x = self.fuse(x)
        x = self.blocks(x)
        return x

class LightMultiScaleSegHead(nn.Module):
    """
    Lightweight Multi-scale Segmentation Head.
    Aggregates hierarchical embeddings from all encoder stages, unifies their channel dimensions, 
    and predicts dense semantic masks for auxiliary task supervision.
    """
    def __init__(self, in_dims, embed_dim):
        super().__init__()
        # Dimension reduction layers for multi-level pyramid features
        self.linear_c4 = nn.Sequential(nn.Conv2d(in_dims[3], embed_dim, 1), LayerNorm2d(embed_dim))
        self.linear_c3 = nn.Sequential(nn.Conv2d(in_dims[2], embed_dim, 1), LayerNorm2d(embed_dim))
        self.linear_c2 = nn.Sequential(nn.Conv2d(in_dims[1], embed_dim, 1), LayerNorm2d(embed_dim))
        self.linear_c1 = nn.Sequential(nn.Conv2d(in_dims[0], embed_dim, 1), LayerNorm2d(embed_dim))

        # Multi-scale feature aggregation and segmentation logit projection
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            LayerNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, 1, kernel_size=1)
        )

    def forward(self, c1, c2, c3, c4):
        size = c1.shape[2:]

        # Spatially align higher-level semantic features to the finest resolution (c1)
        _c = torch.cat([
            F.interpolate(self.linear_c4(c4), size=size, mode='bilinear', align_corners=False),
            F.interpolate(self.linear_c3(c3), size=size, mode='bilinear', align_corners=False),
            F.interpolate(self.linear_c2(c2), size=size, mode='bilinear', align_corners=False),
            self.linear_c1(c1)
        ],dim=1)

        mask_h4 = self.linear_fuse(_c)

        # Restore the prediction to the original image dimension
        out_mask = F.interpolate(mask_h4, scale_factor=4, mode='bilinear', align_corners=False)
        
        return out_mask

class Generator(nn.Module):
    """
    Multi-task Conditional Generator Network.
    Jointly optimizes three interrelated objectives:
    1. Image Translation/Synthesis (Main Task)
    2. Global Pathology/Attribute Classification (Auxiliary Task)
    3. Dense Region-of-Interest Segmentation (Auxiliary Task)
    """
    def __init__(self, in_chans=3, embed_dim=96, depths=[3, 3, 9, 3]):
        super().__init__()

        self.encoder = ConvNeXtEncoder(in_chans, embed_dim, depths)
        
        # Latent bottleneck refinement for deep semantic consolidation
        self.bottle = nn.Sequential(*[
            DecoderResBlock(dim=embed_dim * 8) for _ in range(3)
        ])

        # Auxiliary Classification Branch: Encodes holistic image-level semantics
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),  
            nn.Linear(embed_dim * 8, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        # Auxiliary Segmentation Branch: Captures fine-grained morphological structures
        in_dims =[embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.seg_decoder = LightMultiScaleSegHead(in_dims=in_dims, embed_dim=64)

        # Modulate skip connections to suppress irrelevant background variations
        self.skip3 = SkipRefine(embed_dim * 4) 
        self.skip2 = SkipRefine(embed_dim * 2) 
        self.skip1 = SkipRefine(embed_dim) 

        # Synthesis Decoder mapping latent representations back to pixel space
        self.decoder1 = CNNDecoderStage(in_dim=embed_dim * 8, skip_dim=embed_dim * 4, out_dim=embed_dim * 4, num_blocks=6)
        self.decoder2 = CNNDecoderStage(in_dim=embed_dim * 4, skip_dim=embed_dim * 2, out_dim=embed_dim * 2, num_blocks=4)
        self.decoder3 = CNNDecoderStage(in_dim=embed_dim * 2, skip_dim=embed_dim, out_dim=embed_dim, num_blocks=3)

        # Final reconstruction head yielding standard bounded outputs [-1, 1] via Tanh
        self.final_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            LayerNorm2d(embed_dim // 2), 
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            LayerNorm2d(embed_dim // 4), 
            nn.GELU(),

            DecoderResBlock(dim=embed_dim // 4),
            nn.Conv2d(embed_dim // 4, in_chans, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        # 1. Hierarchical Feature Extraction
        c1, c2, c3, c4 = self.encoder(x)  
        c4 = self.bottle(c4)
        
        # 2. Global Semantic Classification
        pred_class = self.class_head(c4) 
        
        # Detach class probability to prevent segmentation gradients from interfering with classification
        class_prob = torch.sigmoid(pred_class).view(-1, 1, 1, 1).detach()

        # 3. Dense Morphological Segmentation
        pred_mask_raw = self.seg_decoder(c1, c2, c3, c4)

        # Logit-space spatial gating mechanism: Constrains the spatial segmentation activation 
        # heavily conditioned on the overarching global classification probability.
        pred_mask_gated = pred_mask_raw + torch.log(class_prob + 1e-6)

        # 4. Multi-stage Image Synthesis Pipeline
        skip3 = self.skip3(c3)
        skip2 = self.skip2(c2)
        skip1 = self.skip1(c1)

        d1 = self.decoder1(c4, skip3)
        d2 = self.decoder2(d1, skip2)
        d3 = self.decoder3(d2, skip1)

        out = self.final_head(d3)

        return out, pred_class, pred_mask_gated
    
class Discriminator(nn.Module):
    """
    Conditional PatchGAN Discriminator.
    Evaluates the localized structural realism of synthesized target images conditioned on the source domain.
    Spectral Normalization is comprehensively applied to enforce Lipschitz continuity, 
    thereby stabilizing the min-max adversarial training dynamics.
    """
    def __init__(self, input_channels=3, ndf=64, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        
        # Concatenate source (condition) and target images along the channel dimension
        self.stem = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels * 2, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, True)
            )

        layers =[]
        nf_mult = 1
        # Progressively downsample spatial dimensions while expanding feature capacity
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(
                nn.Sequential(
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1)),
                    nn.LeakyReLU(0.2, True)
                )
            )
        nf_mult_final = min(2 ** (n_layers - 1), 8)

        self.path = nn.Sequential(*layers)

        # Predicts a 2D matrix of logits (PatchGAN concept), indicating real/fake probability for overlapping patches
        self.classifier = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult_final, ndf * 8, 4, 1, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1))
        )


    def forward(self, x, condition):
        fused_input = torch.cat([x, condition], dim=1)
        feat_stem = self.stem(fused_input)

        output = self.path(feat_stem)
        pred = self.classifier(output)

        return pred    
    

class MaskedDiscriminator(nn.Module):
    """
    Dual-branch Mask-Aware Conditional PatchGAN Discriminator.
    Simultaneously formulates adversarial evaluation on two levels:
    1. Global Net: Assesses holistic image fidelity.
    2. Regional Net: Imposes stringent structural constraints specifically on mask-identified areas.
    """
    def __init__(self, input_channels=3, ndf=64, n_layers=4):
        super().__init__()
        # Parallel architectures for macro and micro adversarial mapping
        self.global_net = self._build_branch(input_channels * 2, ndf, n_layers)
        self.regional_net = self._build_branch(input_channels * 2, ndf, n_layers)

    def _build_branch(self, in_channels, ndf, n_layers):
        """ Instantiates a standardized PatchGAN branch with Spectral Normalization. """
        layers =[
            spectral_norm(nn.Conv2d(in_channels, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False))
            )
            layers.append(nn.LeakyReLU(0.2, True))

        nf_mult_final = min(2 ** (n_layers - 1), 8)
        layers.append(
            spectral_norm(nn.Conv2d(ndf * nf_mult_final, ndf * 8, 4, 1, 1, bias=False))
        )
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1))
        )
        
        return nn.Sequential(*layers)

    def forward(self, x, condition, mask):
        # Concatenation of generated/real image and semantic conditional prior
        combined_input = torch.cat([x, condition], dim=1)

        # Obtain comprehensive scene-level discrimination
        global_pred = self.global_net(combined_input)

        # Obtain localized feature discrimination
        regional_pred = self.regional_net(combined_input)

        # Spatially align the target segmentation mask to match PatchGAN's output resolution 
        # using nearest-neighbor interpolation to preserve discrete boundary definitions.
        mask_downsampled = F.interpolate(mask, size=regional_pred.shape[-2:], mode='nearest')

        return {
            "global_pred": global_pred,
            "regional_pred": regional_pred, 
            "mask_downsampled": mask_downsampled
        }