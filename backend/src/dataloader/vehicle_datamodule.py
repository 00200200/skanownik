import os
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class VehicleDataset(Dataset):
    def __init__(self, data: List[Tuple[Path, int]], transform=None):
        """
        Args:
            data: List of tuples (image_path, label)
            transform: Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # Read image using OpenCV
        # The images are BMP.
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

class VehicleDataModule(LightningDataModule):
    def __init__(
        self, 
        data_dir: str = "data", 
        batch_size: int = 32, 
        num_workers: int = 4,
        train_val_split: float = 0.8,
        image_size: Tuple[int, int] = (224, 224)
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self, stage: Optional[str] = None):
        if self.dataset_train is not None:
            return

        # Collect all data
        # Structure: data/{brudne,czyste}/{osobowe,dostawcze}/{timestamp}/*.bmp
        
        data_samples = []
        
        # Label mapping: osobowe -> 0, dostawcze -> 1
        label_map = {
            "osobowe": 0,
            "dostawcze": 1
        }
        
        # Iterate over brudne and czyste
        for quality in ["brudne", "czyste"]:
            quality_path = self.data_dir / quality
            if not quality_path.exists():
                print(f"Warning: {quality_path} does not exist")
                continue
                
            for class_name, label in label_map.items():
                class_path = quality_path / class_name
                if not class_path.exists():
                    continue
                    
                # Iterate over timestamp folders
                for folder in class_path.iterdir():
                    if not folder.is_dir():
                        continue
                        
                    # Take only 'czarno' bmp files
                    bmp_files = list(folder.glob("*czarno.bmp"))
                    
                    for bmp_file in bmp_files:
                        data_samples.append((bmp_file, label))

        print(f"Total samples found: {len(data_samples)}")
        
        # Create full dataset
        full_dataset = VehicleDataset(data_samples, transform=self.transform)
        
        # Split into Train, Val, Test (70%, 15%, 15%)
        total_size = len(full_dataset)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.15)
        test_size = total_size - train_size - val_size
        
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(
            full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

if __name__ == "__main__":
    # Test the datamodule
    dm = VehicleDataModule(data_dir=str(Path(__file__).parent.parent.parent / "data"))
    dm.setup()
    
    print(f"Train set size: {len(dm.dataset_train)}")
    print(f"Val set size: {len(dm.dataset_val)}")
    
    print(f"Test set size: {len(dm.dataset_test)}")
    # Get a batch
    loader = dm.train_dataloader()
    images, labels = next(iter(loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
