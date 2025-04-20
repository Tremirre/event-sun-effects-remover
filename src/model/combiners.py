import torch


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


COMBINERS = {
    "simple_concat": SimpleConcatCombiner,
}


def get_combiner(name: str, **kwargs) -> BaseCombiner:
    """
    Returns the combiner class based on the name.
    """
    if name not in COMBINERS:
        raise ValueError(f"Combiner {name} not found.")
    return COMBINERS[name](**kwargs)
