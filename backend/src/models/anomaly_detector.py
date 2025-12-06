import os
from typing import Any

import joblib
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.neighbors import NearestNeighbors
from torchvision import models


class AnomalyDetectorPL(LightningModule):
    """Anomaly Detector using PyTorch Lightning.

    Uses a pre-trained ResNet backbone to extract features from image patches
    and a KNN memory bank to detect anomalies based on distance to normal features.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        memory_bank_size: int = 10000,
        use_coords: bool = False,
        knn_path: str = "trained_models_anomaly/knn_model.pkl",
    ) -> None:
        """Initialize the AnomalyDetectorPL.

        Args:
            backbone_name: Name of the ResNet backbone to use.
            memory_bank_size: Maximum size of the memory bank.
            use_coords: Whether to include spatial coordinates in features.
            knn_path: Path to save/load the KNN model.
        """
        super().__init__()
        self.save_hyperparameters()
        self.use_coords = use_coords
        self.knn_path = knn_path

        # 1. Backbone
        # Load pre-trained ResNet
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone.fc = nn.Identity()
        elif backbone_name == "resnet34":
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.backbone.fc = nn.Identity()
        elif backbone_name == "wide_resnet50_2":
            self.backbone = models.wide_resnet50_2(
                weights=models.Wide_ResNet50_2_Weights.DEFAULT
            )
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Freeze weights
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Memory Bank
        self.memory_bank: list[torch.Tensor] = []
        self.knn: NearestNeighbors | None = None
        self.memory_bank_size = memory_bank_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.

        Args:
            x: Input tensor.

        Returns:
            Extracted features.
        """
        # Extract features
        # ResNet forward: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2
        # -> layer3 -> layer4 -> avgpool -> fc
        # We want the output of avgpool (which is a vector per image/patch)

        # Since we replaced fc with Identity, calling self.backbone(x)
        # returns the pooled features.
        features = self.backbone(x)
        return features

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        # Batch is now a dict
        patches = batch["patch"]

        features = self(patches)  # [B, D]

        if self.use_coords:
            xs = batch["x"]
            ys = batch["y"]
            orig_hs = batch["orig_h"]
            orig_ws = batch["orig_w"]

            # Normalize coordinates
            # We assume all images are roughly same size or we normalize to [0, 1]
            # Add a weight factor to coordinates to balance with visual features
            coord_weight = 5.0

            xs_norm = (xs.float() / orig_ws.float()) * coord_weight
            ys_norm = (ys.float() / orig_hs.float()) * coord_weight

            # Concatenate [B, D] with [B, 1] and [B, 1] -> [B, D+2]
            features_to_save = torch.cat(
                [
                    features,
                    xs_norm.unsqueeze(1).to(features.device),
                    ys_norm.unsqueeze(1).to(features.device),
                ],
                dim=1,
            )
        else:
            features_to_save = features

        # Detach and move to CPU to save memory
        if len(self.memory_bank) == 0:
            print(
                f"Feature stats: Mean={features.mean()}, Std={features.std()}, "
                f"Max={features.max()}, Min={features.min()}"
            )
        self.memory_bank.append(features_to_save.detach().cpu())

        return None  # No loss to optimize

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch."""
        # 1. Aggregate Memory Bank
        if not self.memory_bank:
            print("Warning: Memory bank is empty.")
            return

        full_bank = torch.cat(self.memory_bank, dim=0)  # [N, D]
        print(f"Full memory bank shape: {full_bank.shape}")

        # 2. Downsample (Random Sampling for simplicity)
        if full_bank.shape[0] > self.memory_bank_size:
            indices = torch.randperm(full_bank.shape[0])[: self.memory_bank_size]
            full_bank = full_bank[indices]
            print(f"Downsampled memory bank to: {full_bank.shape}")

        # 3. Fit KNN
        # Convert to numpy
        x_np = full_bank.numpy()

        print("Fitting KNN model...")
        self.knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        self.knn.fit(x_np)

        # Save KNN model
        os.makedirs(os.path.dirname(self.knn_path), exist_ok=True)
        joblib.dump(self.knn, self.knn_path)
        print(f"KNN model saved to {self.knn_path}")

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """Prediction step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Dictionary containing scores and metadata.
        """
        # Batch is a dict with 'patch', 'x', 'y', etc.
        patches = batch["patch"]

        features = self(patches)  # [B, D]

        if self.use_coords:
            xs = batch["x"]
            ys = batch["y"]
            orig_hs = batch["orig_h"]
            orig_ws = batch["orig_w"]

            # Normalize coordinates
            coord_weight = 10.0
            xs_norm = (xs.float() / orig_ws.float()) * coord_weight
            ys_norm = (ys.float() / orig_hs.float()) * coord_weight

            features_to_query = torch.cat(
                [
                    features,
                    xs_norm.unsqueeze(1).to(features.device),
                    ys_norm.unsqueeze(1).to(features.device),
                ],
                dim=1,
            )
        else:
            features_to_query = features

        features_np = features_to_query.detach().cpu().numpy()

        if self.knn is None:
            # Try loading
            try:
                self.knn = joblib.load(self.knn_path)
            except Exception as e:
                raise RuntimeError(
                    f"KNN model not found at {self.knn_path}. Train the model first."
                ) from e

        # Query KNN
        distances, _ = self.knn.kneighbors(features_np)

        # Anomaly score is the distance to the nearest clean patch
        scores = distances[:, 0]  # [B]

        return {
            "scores": scores,
            "x": batch["x"].cpu().numpy(),
            "y": batch["y"].cpu().numpy(),
            "img_idx": batch["img_idx"].cpu().numpy(),
            "img_path": batch["img_path"],
            "label": batch["label"].cpu().numpy(),
        }

    def configure_optimizers(self) -> None:
        """Configure optimizers."""
        return None  # No optimizer needed
