import torch


class SimpleConcatCombiner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, artifact_map: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, artifact_map], dim=1)


COMBINERS = {
    "simple_concat": SimpleConcatCombiner,
}
