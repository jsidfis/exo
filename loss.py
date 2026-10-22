"""Mean squared error over originally observed joint targets."""
import torch


def supervised_mse(prediction, target):
    valid = torch.isfinite(target)
    if not valid.any():
        raise ValueError('No finite targets in supervised batch')
    return (prediction[valid] - target[valid]).square().mean()
