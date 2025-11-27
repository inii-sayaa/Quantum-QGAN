
import torch
import math


def sample_real_data(n_samples: int, radius: float = 1.0, noise_std: float = 0.05) -> torch.Tensor:

    angles = 2 * math.pi * torch.rand(n_samples)
    radial_noise = 1.0 + noise_std * torch.randn(n_samples)
    x = radius * radial_noise * torch.cos(angles)
    y = radius * radial_noise * torch.sin(angles)
    return torch.stack((x, y), dim=1)
