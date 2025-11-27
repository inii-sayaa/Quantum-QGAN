"""
Quantum GAN Package

A modular implementation of Quantum Generative Adversarial Networks.
"""
from .config import get_default_config
from .models import QuantumGenerator, ClassicalDiscriminator
from .training import train_qgan, set_global_seeds
from .utils import sample_real_data, compute_mmd
from .visualization import plot_results_basic, plot_results_enhanced

__version__ = "0.1.0"

__all__ = [
    'get_default_config',
    'QuantumGenerator',
    'ClassicalDiscriminator',
    'train_qgan',
    'set_global_seeds',
    'sample_real_data',
    'compute_mmd',
    'plot_results_basic',
    'plot_results_enhanced',
]
