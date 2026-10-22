"""Seed Python, NumPy and PyTorch in the training initialization order."""
import random

import numpy as np
import torch


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(seed)
