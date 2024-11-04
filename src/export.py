import torch

from src import const
from src.model import unet

if __name__ == "__main__":
    model = unet.UNet(3)
    # export to onnx
    model.eval()
    x = torch.randn(1, 5, const.IMG_HEIGHT, const.IMG_WIDTH)
    torch.onnx.export(model, x, "model.onnx", verbose=True)
