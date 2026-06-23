import os
import cv2
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

# Suppress library update warnings to ensure clean standard output streams in cluster environments
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


class MedicalDataset(Dataset):
    """
    Multi-modal Paired Histopathology Dataset.
    Manages the deterministic loading, alignment, and spatial-preserving augmentation 
    of Source (H&E), Target (IHC), and corresponding semantic masks.
    """
    def __init__(self, root_A, root_B, mask_root_B, img_size, is_train):
        self.root_A = root_A
        self.root_B = root_B
        self.mask_root_B = mask_root_B
        self.img_size = img_size
        self.is_train = is_train

        # Zero-centered normalization mapping pixel intensities to the [-1, 1] manifold, 
        # which is mathematically optimal for Tanh-activated adversarial optimization.
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.all_files = sorted([f for f in os.listdir(root_A) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        if self.is_train:
            # Spatially-synchronized morphological augmentations.
            # Albumentations is explicitly utilized to ensure that the source domain, 
            # target domain, and dense masks undergo mathematically identical transformations,
            # thereby preserving the strict pixel-to-pixel spatial correspondence.
            self.aug_transform = A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(-0.02, 0.02),
                    rotate=(-5, 5),
                    p=0.3
                ),
                # Elastic deformations simulate natural biological tissue variations 
                # (e.g., cutting artifacts, stretching) to enhance model robustness.
                A.ElasticTransform(alpha=120, sigma=12, p=0.2),
            ], 
            additional_targets={'image_B': 'image', 'mask_B': 'mask'})
        else:
            # Deterministic evaluation pipeline (strictly resizing without stochasticity)
            self.aug_transform = A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC)
            ], additional_targets={'image_B': 'image'})

    def _get_mask_path(self, filename):
        name_no_ext = os.path.splitext(filename)[0]
        return os.path.join(self.mask_root_B, name_no_ext + '.png')

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        """
        Retrieves and processes a multi-modal data tuple for a single empirical instance.
        """
        filename = self.all_files[index]
        path_A = os.path.join(self.root_A, filename)
        path_B = os.path.join(self.root_B, filename)

        img_A = cv2.imread(path_A)
        img_B = cv2.imread(path_B)

        if img_A is None: raise FileNotFoundError(f"Missing: {path_A}")
        if img_B is None: raise FileNotFoundError(f"Missing: {path_B}")

        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        if self.is_train:
            mask_path = self._get_mask_path(filename)
            mask_B = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask_B is None: raise FileNotFoundError(f"Missing Mask: {mask_path}")

            # Apply synchronized spatial augmentations across all modalities
            augmented = self.aug_transform(image=img_A, image_B=img_B, mask_B=mask_B)
            img_A = augmented['image']
            img_B = augmented['image_B']
            mask_B = augmented['mask_B']

            pos_pixels = cv2.countNonZero(mask_B)

            # Dynamic Weak-Label Extraction: 
            # Infers the global pathology status (e.g., HER2 3+ expression) based on the 
            # presence of positive staining pixels within the dense semantic mask.
            is_3plus = 1.0 if pos_pixels > 0.0 else 0.0

            mask_tensor = torch.from_numpy(mask_B).unsqueeze(0).float() / 255.0

            return {
                'A': self.base_transform(img_A),
                'B': self.base_transform(img_B),
                'B_mask': mask_tensor,
                'label_3plus': torch.tensor([is_3plus], dtype=torch.float32)
            }

        else:
            augmented = self.aug_transform(image=img_A, image_B=img_B)
            img_A, img_B = augmented['image'], augmented['image_B']

            return {
                'A': self.base_transform(img_A),
                'B': self.base_transform(img_B),
                'A_filename': os.path.splitext(filename)[0]
            }


def collate_fn(batch):
    """
    Custom batch collation mechanism.
    Dynamically constructs heterogeneous batched tensors based on the current phase 
    (training with auxiliary masks vs. purely inference).
    """
    imgs_A = torch.stack([item['A'] for item in batch], dim=0)
    imgs_B = torch.stack([item['B'] for item in batch], dim=0)

    output = {'A': imgs_A, 'B': imgs_B}

    # Conditionally aggregate training-specific dense supervision and weak labels
    if 'B_mask' in batch[0]:
        output['B_mask'] = torch.stack([item['B_mask'] for item in batch], dim=0)
        output['label_3plus'] = torch.stack([item['label_3plus'] for item in batch], dim=0)

    # Retain spatial tracking metadata during inference for result serialization
    elif 'A_filename' in batch[0]:
        output['A_filename'] = [item['A_filename'] for item in batch]

    return output


def get_loader(root_A, root_B, mask_root_B, batch_size, img_size, is_train,
               distributed=False, rank=0, world_size=1, num_workers=4):
    """
    Instantiates the multi-processing data loader pipeline.
    Seamlessly integrates DistributedSampler for non-overlapping empirical distribution 
    sampling across multi-GPU DDP topologies.
    """
    dataset = MedicalDataset(root_A, root_B, mask_root_B, img_size, is_train)
    sampler = None
    shuffle = True if is_train else False

    if distributed:
        # Enforce distinct, non-overlapping dataset partitions across distributed nodes
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True, # Accelerates host-to-device (CPU-to-GPU) data transfers
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=(num_workers>0)
    )

    return loader, sampler