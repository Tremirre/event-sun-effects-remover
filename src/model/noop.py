import pytorch_lightning as pl
import torch


class NoOp(pl.LightningModule):
    """
    Debug model that does nothing for testing the pipeline.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :3]

    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0)

    def validation_step(self, batch, batch_idx):
        return torch.tensor(0.0)

    def test_step(self, batch, batch_idx):
        return torch.tensor(0.0)

    def configure_optimizers(self):
        return None

    def backward(
        self, loss: torch.Tensor, *args: torch.Any, **kwargs: torch.Any
    ) -> None:
        pass
