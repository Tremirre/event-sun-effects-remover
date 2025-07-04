import torch
import torchvision.transforms as TV

from .models import ConvBlock

COMBINER_IN_CHANNELS = 5


class BaseCombiner(torch.nn.Module):
    def get_output_channels(self) -> int:
        """
        Returns the number of output channels for the combiner.
        This is used to determine the number of input channels for the inpainter.
        """
        raise NotImplementedError("Combiner must implement get_output_channels method.")

    def forward(self, x: torch.Tensor, artifact_map: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Combiner must implement forward method.")


class SimpleConcatCombiner(BaseCombiner):
    def get_output_channels(self) -> int:
        return 5

    def forward(self, x: torch.Tensor, artifact_map: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, artifact_map], dim=1)


class MaskedRemovalCombiner(BaseCombiner):
    def __init__(
        self,
        binarize: bool = False,
        yuv_interpolation: bool = False,
        soft_factor: float = 0,
        threshold: float = 0.0,
    ):
        super().__init__()
        self.binarize = binarize
        self.yuv_interpolation = yuv_interpolation
        self.kernel_size = soft_factor * 2 + 1
        self.soft_factor = soft_factor
        self.threshold = threshold
        self.gaussian_blur = (
            TV.GaussianBlur(self.kernel_size, sigma=self.soft_factor)
            if soft_factor > 0
            else None
        )

    def get_output_channels(self) -> int:
        return 5  # 3 masked BGR channels + 1 event reconstruction channel + 1 mask

    def _bgr_to_yuv(self, bgr: torch.Tensor) -> torch.Tensor:
        transform = torch.tensor(
            [
                [0.098, 0.504, 0.257],  # Y (from BGR)
                [0.439, -0.291, -0.148],  # U (from BGR)
                [-0.071, -0.368, 0.439],  # V (from BGR)
            ],
            device=bgr.device,
        )

        bias = torch.tensor([16, 128, 128], device=bgr.device) / 255.0
        b, c, h, w = bgr.shape
        bgr_reshaped = bgr.permute(0, 2, 3, 1).reshape(b * h * w, 3)
        yuv_reshaped = torch.matmul(bgr_reshaped, transform.t()) + bias
        return yuv_reshaped.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    def _yuv_to_bgr(self, yuv: torch.Tensor) -> torch.Tensor:
        bias = torch.tensor([16, 128, 128], device=yuv.device) / 255.0
        transform = torch.tensor(
            [
                [1.164, 0.000, 1.596],  # B
                [1.164, -0.392, -0.813],  # G
                [1.164, 2.017, 0.000],  # R
            ],
            device=yuv.device,
        )
        b, c, h, w = yuv.shape
        yuv_reshaped = yuv.permute(0, 2, 3, 1).reshape(b * h * w, 3)
        yuv_biased = yuv_reshaped - bias
        bgr_reshaped = torch.matmul(yuv_biased, transform.t())
        bgr_reshaped = torch.clamp(bgr_reshaped, 0, 1)
        return bgr_reshaped.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor, artifact_map: torch.Tensor) -> torch.Tensor:
        bgr_channels = x[:, :3, :, :]
        event_reconstruction = x[:, 3:4, :, :]
        mask = artifact_map
        if self.binarize:
            mask = (mask > self.threshold).float()
            if self.gaussian_blur is not None:
                mask = self.gaussian_blur(mask)

        inverse_mask = 1 - mask

        if self.yuv_interpolation:
            yuv_channels = self._bgr_to_yuv(bgr_channels)
            y_channel = yuv_channels[:, 0:1, :, :]
            uv_channels = yuv_channels[:, 1:, :, :]
            masked_y = y_channel * inverse_mask
            masked_yuv = torch.cat([masked_y, uv_channels], dim=1)
            masked_bgr = self._yuv_to_bgr(masked_yuv)
        else:
            masked_bgr = bgr_channels * inverse_mask
        return torch.cat([masked_bgr, event_reconstruction, artifact_map], dim=1)


class ConvolutionalCombiner(BaseCombiner):
    def __init__(self, out_channels: int, depth: int, kernel_size: int = 3):
        super().__init__()
        self.conv = ConvBlock(
            COMBINER_IN_CHANNELS,
            out_channels,
            depth,
            kernel_size,
            activation_func="sigmoid",
        )

    def get_output_channels(self) -> int:
        return self.conv.out_channels

    def forward(self, x: torch.Tensor, artifact_map: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, artifact_map], dim=1)
        return self.conv(x)


class EventConsideringCombiner(BaseCombiner):
    def __init__(self, blending_factor: float = 0.5) -> None:
        super().__init__()
        self.bgr_to_gray = TV.Grayscale(num_output_channels=1)
        self.blending_factor = blending_factor

    def get_output_channels(self) -> int:
        return 5

    def forward(self, x: torch.Tensor, artifact_map: torch.Tensor) -> torch.Tensor:
        bgr_channels = x[:, :3, :, :]
        event_reconstruction = x[:, 3:4, :, :]
        mask = artifact_map
        gs = self.bgr_to_gray(bgr_channels)
        diff = torch.relu(gs - event_reconstruction)

        diff_min = diff.amin(dim=(2, 3), keepdim=True)
        diff_max = diff.amax(dim=(2, 3), keepdim=True)
        diff = (diff - diff_min) / (diff_max - diff_min + 1e-6)
        adjusted_mask = mask + self.blending_factor * diff
        adjusted_mask = torch.clamp(adjusted_mask, 0, 1)
        return torch.cat((bgr_channels, event_reconstruction, adjusted_mask), dim=1)


class EventMaskedRemovalCombiner(BaseCombiner):
    def __init__(
        self,
        binarize: bool = False,
        yuv_interpolation: bool = False,
        soft_factor: float = 0,
        blending_factor: float = 0.5,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.masked_removal = MaskedRemovalCombiner(
            binarize=binarize,
            yuv_interpolation=yuv_interpolation,
            soft_factor=soft_factor,
            threshold=threshold,
        )
        self.event_considering = EventConsideringCombiner(
            blending_factor=blending_factor
        )

    def get_output_channels(self) -> int:
        return self.masked_removal.get_output_channels()

    def forward(self, x: torch.Tensor, artifact_map: torch.Tensor) -> torch.Tensor:
        event_combined = self.event_considering(x, artifact_map)
        return self.masked_removal(
            event_combined[:, :4, :, :], event_combined[:, 4, :, :]
        )


COMBINERS = {
    "simple_concat": SimpleConcatCombiner,
    "masked_removal": MaskedRemovalCombiner,
    "convolutional": ConvolutionalCombiner,
    "event_considering": EventConsideringCombiner,
    "event_masked_removal": EventMaskedRemovalCombiner,
}


def get_combiner(name: str, **kwargs) -> BaseCombiner:
    """
    Returns the combiner class based on the name.
    """
    if name not in COMBINERS:
        raise ValueError(f"Combiner {name} not found.")
    return COMBINERS[name](**kwargs)
