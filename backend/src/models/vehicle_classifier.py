import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchvision import models


class VehicleClassifier(LightningModule):
    """Vehicle Classifier using PyTorch Lightning.

    Classifies vehicles into two categories: 'osobowe' (0) and 'dostawcze' (1).
    Uses a pre-trained ResNet18 backbone.
    """

    def __init__(self, learning_rate: float = 1e-3) -> None:
        """Initialize the VehicleClassifier.

        Args:
            learning_rate: Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        # Use a pre-trained ResNet18
        # We need to replace the final fully connected layer to output 2 classes
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # ResNet18 fc layer input features is 512
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: osobowe (0), dostawcze (1)

        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Logits.
        """
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Input batch (x, y).
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Input batch (x, y).
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch: Input batch (x, y).
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers.

        Returns:
            Optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    # Test model instantiation
    model = VehicleClassifier()
    print("Model created successfully")

    # Test forward pass with dummy data
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
