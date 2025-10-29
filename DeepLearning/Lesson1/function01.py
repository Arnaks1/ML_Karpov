import torch


def function01(tensor: torch.Tensor, count_over: str) -> torch.Tensor:
    if count_over == 'columns':
        return tensor.mean(dim=0)  # Среднее по колонкам
    elif count_over == 'rows':
        return tensor.mean(dim=1)  # Среднее по рядам
    else:
        raise ValueError("count_over must be either 'columns' or 'rows'")


x = torch.Tensor([[1, 2],
                  [3, 4],
                  [5, 6]])
print(function01(x, 'columns'))
