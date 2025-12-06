import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from src.utils.image_utils import tile_image, align_to_reference

class XRayDataset(Dataset):
    def __init__(
        self, 
        image_paths: List[Path], 
        transform=None, 
        patch_size: int = 224, 
        stride: int = 112,
        mode: str = 'train'
    ):
        """
        Args:
            image_paths: List of paths to images.
            transform: PyTorch transforms.
            patch_size: Size of patches.
            stride: Stride for tiling.
            mode: 'train' (returns flattened patches) or 'predict' (returns patches grouped by image info).
        """
        self.image_paths = image_paths
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        
        self.samples = []
        self._prepare_data()

    def _prepare_data(self):
        """
        Loads images, tiles them, and prepares the sample list.
        For 'train' mode: samples are individual patches.
        For 'predict' mode: samples are individual patches but we keep track of image index and coords.
        """
        print(f"Preparing data for mode: {self.mode}...")
        for img_idx, img_path in enumerate(self.image_paths):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Try to find the corresponding "blue/metal" image and blend
            # Current path: .../ID czarno.bmp -> Target path: .../ID.bmp
            path_str = str(img_path)
            if " czarno.bmp" in path_str:
                other_path_str = path_str.replace(" czarno.bmp", ".bmp")
                if os.path.exists(other_path_str):
                    other_img = cv2.imread(other_path_str)
                    if other_img is not None:
                        if img.shape == other_img.shape:
                            # Blend 90/10
                            img = cv2.addWeighted(img, 0.9, other_img, 0.1, 0)
                        else:
                            print(f"Warning: Shape mismatch for {img_path} vs {other_path_str}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Align (Placeholder)
            img = align_to_reference(img)
            
            # Tile
            # For training, we filter background aggressively.
            # For testing, we might want to keep everything or filter less? 
            # Let's stick to filtering to avoid false positives on background noise.
            patches_with_coords = tile_image(
                img, 
                patch_size=self.patch_size, 
                stride=self.stride,
                filter_background=True
            )
            
            for patch, x, y in patches_with_coords:
                # Determine label from path
                label = 1 if "brudne" in str(img_path) else 0
                
                self.samples.append({
                    'patch': patch,
                    'x': x,
                    'y': y,
                    'img_idx': img_idx,
                    'img_path': str(img_path),
                    'orig_h': img.shape[0],
                    'orig_w': img.shape[1],
                    'label': label
                })
        print(f"Total patches generated: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        patch = sample['patch']
        
        if self.transform:
            patch = self.transform(patch)
            
        if self.mode == 'train':
            # Just return the patch tensor
            return patch
        else:
            # Return patch and metadata for reconstruction
            return {
                'patch': patch,
                'x': sample['x'],
                'y': sample['y'],
                'img_idx': sample['img_idx'],
                'img_path': sample['img_path'],
                'orig_h': sample['orig_h'],
                'orig_w': sample['orig_w'],
                'label': sample['label']
            }

class XRayDataModule(LightningDataModule):
    def __init__(
        self, 
        data_dir: str = "data", 
        vehicle_type: str = "dostawcze",
        batch_size: int = 32, 
        num_workers: int = 4,
        patch_size: int = 224,
        stride: int = 112
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.vehicle_type = vehicle_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.stride = stride
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.patch_size, self.patch_size)), # Ensure size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.train_dataset = None
        self.predict_dataset = None

    def setup(self, stage: Optional[str] = None):
        # Train data: "czyste" only
        train_dir = self.data_dir / "czyste" / self.vehicle_type
        all_clean_files = []
        if train_dir.exists():
            # Recursively find all .bmp files
            all_clean_files = list(train_dir.glob("**/*.bmp"))
            # Filter for 'czarno' if needed
            all_clean_files = [f for f in all_clean_files if "czarno" in f.name]
            
        # Deterministic split: 90% Train, 10% Validation (Test)
        import random
        random.seed(42)
        random.shuffle(all_clean_files)
        
        split_idx = int(len(all_clean_files) * 0.9)
        train_clean_files = all_clean_files[:split_idx]
        val_clean_files = all_clean_files[split_idx:]
        
        print(f"Clean images: {len(all_clean_files)}. Split: {len(train_clean_files)} train, {len(val_clean_files)} val (reserved).")
        
        if stage == 'fit' or stage is None:
            self.train_dataset = XRayDataset(
                train_clean_files, 
                transform=self.transform,
                patch_size=self.patch_size,
                stride=self.stride,
                mode='train'
            )
            
        # Predict/Test data: Reserved "czyste" + "brudne"
        if stage == 'predict' or stage == 'test':
            test_files = []
            
            # Add reserved clean images for validation
            test_files.extend(val_clean_files)
            
            # Add dirty images
            dirty_dir = self.data_dir / "brudne" / self.vehicle_type
            if dirty_dir.exists():
                dirty_files = list(dirty_dir.glob("**/*.bmp"))
                dirty_files = [f for f in dirty_files if "czarno" in f.name]
                test_files.extend(dirty_files)
            
            self.predict_dataset = XRayDataset(
                test_files,
                transform=self.transform,
                patch_size=self.patch_size,
                stride=self.stride,
                mode='predict'
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
