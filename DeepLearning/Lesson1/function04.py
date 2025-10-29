import torch
import torch.nn as nn


def function04(x: torch.Tensor, y: torch.Tensor) -> nn.Linear:
    fc = nn.Linear(x.shape[1], 1, bias=False)

    lr = 1e-2
    n_epochs = 10000

    for epoch in range(n_epochs):
        y_pred = fc(x).squeeze()

        mse_loss = ((y_pred - y)**2).mean()

        if mse_loss.item() < 0.3:
            break

        mse_loss.backward()

        with torch.no_grad():
            for param in fc.parameters():
                param -= lr * param.grad
                param.grad.zero_()

    return fc
