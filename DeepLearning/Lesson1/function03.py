import torch


def function02(dataset: torch.Tensor) -> torch.Tensor:
    weights = torch.rand(dataset.shape[1], dtype=torch.float32, requires_grad=True)
    return weights


def function03(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    w = function02(x)

    lr = 1e-2
    n_epochs = 1000

    for epoch in range(n_epochs):
        y_pred = x @ w

        mse_loss = ((y_pred - y)**2).mean()
        if mse_loss.item() < 1:
            break

        mse_loss.backward()

        with torch.no_grad():
            w -= lr * w.grad

            w.grad.zero_()

    return w
