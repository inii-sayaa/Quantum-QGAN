import torch
import math

def sample_real_data(n_samples: int, radius: float = 1.0, noise_std: float = 0.05) -> torch.Tensor:

    angles = 2 * math.pi * torch.rand(n_samples)
    radial_noise = 1.0 + noise_std * torch.randn(n_samples)
    x = radius * radial_noise * torch.cos(angles)
    y = radius * radial_noise * torch.sin(angles)
    return torch.stack((x, y), dim=1)

def compute_mmd(real_samples: torch.Tensor, fake_samples: torch.Tensor, sigma: float = 1.0) -> float:
    
    def kernel(x, y):
        dist_sq = torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2))
    
    real_kernel = kernel(real_samples, real_samples)
    fake_kernel = kernel(fake_samples, fake_samples)
    cross_kernel = kernel(real_samples, fake_samples)
    
    mmd = real_kernel.mean() + fake_kernel.mean() - 2 * cross_kernel.mean()
    return mmd.item()
