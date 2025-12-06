import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchvision import models
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib
import os

class AnomalyDetectorPL(LightningModule):
    def __init__(self, backbone_name='resnet18', memory_bank_size=10000):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Backbone
        # Load pre-trained ResNet
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone_name == 'wide_resnet50_2':
            self.backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
            
        # Remove FC layers (we only need feature extractor)
        # We will use the output of the average pooling layer (before fc)
        # Or we can extract intermediate features.
        # Simplified PatchCore: Use the final embedding after GAP.
        self.backbone.fc = nn.Identity()
        
        # Freeze weights
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Memory Bank
        self.memory_bank = []
        self.knn = None
        self.memory_bank_size = memory_bank_size

    def forward(self, x):
        # Extract features
        # ResNet forward: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        # We want the output of avgpool (which is a vector per image/patch)
        
        # Since we replaced fc with Identity, calling self.backbone(x) returns the pooled features.
        features = self.backbone(x)
        return features

    def training_step(self, batch, batch_idx):
        # Batch is just a tensor of patches [B, C, H, W]
        features = self(batch) # [B, D]
        
        # Detach and move to CPU to save memory
        self.memory_bank.append(features.detach().cpu())
        
        return None # No loss to optimize

    def on_train_epoch_end(self):
        # 1. Aggregate Memory Bank
        if not self.memory_bank:
            print("Warning: Memory bank is empty.")
            return
            
        full_bank = torch.cat(self.memory_bank, dim=0) # [N, D]
        print(f"Full memory bank shape: {full_bank.shape}")
        
        # 2. Downsample (Random Sampling for simplicity)
        if full_bank.shape[0] > self.memory_bank_size:
            indices = torch.randperm(full_bank.shape[0])[:self.memory_bank_size]
            full_bank = full_bank[indices]
            print(f"Downsampled memory bank to: {full_bank.shape}")
            
        # 3. Fit KNN
        # Convert to numpy
        X = full_bank.numpy()
        
        print("Fitting KNN model...")
        self.knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        self.knn.fit(X)
        
        # Save KNN model
        os.makedirs("trained_models_anomaly", exist_ok=True)
        joblib.dump(self.knn, "trained_models_anomaly/knn_model.pkl")
        print("KNN model saved.")

    def predict_step(self, batch, batch_idx):
        # Batch is a dict with 'patch', 'x', 'y', etc.
        patches = batch['patch']
        
        features = self(patches) # [B, D]
        features_np = features.detach().cpu().numpy()
        
        if self.knn is None:
            # Try loading
            try:
                self.knn = joblib.load("trained_models_anomaly/knn_model.pkl")
            except:
                raise RuntimeError("KNN model not found. Train the model first.")
        
        # Query KNN
        distances, _ = self.knn.kneighbors(features_np)
        
        # Anomaly score is the distance to the nearest clean patch
        scores = distances[:, 0] # [B]
        
        return {
            'scores': scores,
            'x': batch['x'].cpu().numpy(),
            'y': batch['y'].cpu().numpy(),
            'img_idx': batch['img_idx'].cpu().numpy(),
            'img_path': batch['img_path'],
            'label': batch['label'].cpu().numpy()
        }

    def configure_optimizers(self):
        return None # No optimizer needed
