
import torch

def get_default_config():
    
    return {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "latent_dim": 2,
        "n_qubits": 3,
        "n_layers": 4,
        "scale_factor": 1.2,
        "dataset_size": 2048,
        "batch_size": 128,
        "num_epochs": 300,
        "lr_G": 2e-2,
        "lr_D": 1e-3,
        "weight_decay": 1e-5,
        "gradient_clip": 1.0,
    }
