import torch


def function02(dataset: torch.Tensor) -> torch.Tensor:
    weights = torch.rand(dataset.shape[1], dtype=torch.float32, requires_grad=True)
    return weights
