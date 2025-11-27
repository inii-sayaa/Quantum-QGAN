# QGAN Project Structure

## Directory Tree

```
QGAN/
│
├── 📄 main.py                    # Main entry point untuk training
├── 📄 README.md                  # Dokumentasi project
├── 📄 MIGRATION_GUIDE.md         # Guide untuk migrasi struktur
├── 📄 requirements.txt           # Python dependencies
│
├── 📁 src/                       # Source code package
│   ├── 📄 __init__.py            # Package initialization & public API
│   │
│   ├── 📁 config/                # Configuration management
│   │   ├── 📄 __init__.py
│   │   └── 📄 settings.py        # Default configurations
│   │
│   ├── 📁 models/                # Neural network models
│   │   ├── 📄 __init__.py
│   │   ├── 📄 quantum_generator.py         # Quantum VQC Generator
│   │   └── 📄 classical_discriminator.py   # Classical MLP Discriminator
│   │
│   ├── 📁 training/              # Training logic
│   │   ├── 📄 __init__.py
│   │   └── 📄 trainer.py         # Main training loop & utilities
│   │
│   ├── 📁 utils/                 # Utility functions
│   │   ├── 📄 __init__.py
│   │   ├── 📄 data.py            # Data sampling utilities
│   │   └── 📄 metrics.py         # Evaluation metrics (MMD, etc.)
│   │
│   └── 📁 visualization/         # Plotting and visualization
│       ├── 📄 __init__.py
│       └── 📄 plotting.py        # Plot generation functions
│
├── 📁 data/                      # Data directory (if needed)
│
├── 📁 output/                    # Output directory
│   ├── qgan_analysis.png         # Training analysis plots
│   ├── qgan_enhanced.png         # Enhanced visualization
│   └── qgan_checkpoint.pth       # Model checkpoint
│
└── 📁 venv/                      # Virtual environment (git-ignored)
```

## Module Overview

### 1 **config/** - Configuration Management
- Centralized configuration settings
- Easy to modify hyperparameters
- Reusable across different experiments

### 2 **models/** - Neural Network Models
- `quantum_generator.py`: Variational Quantum Circuit (VQC)
- `classical_discriminator.py`: Classical neural network
- Clean separation of quantum and classical components

### 3 **training/** - Training Pipeline
- Complete training loop implementation
- Seed management for reproducibility
- Optimizer and scheduler setup

### 4 **utils/** - Utility Functions
- `data.py`: Data generation and sampling
- `metrics.py`: Evaluation metrics (MMD)
- Reusable helper functions

### 5 **visualization/** - Plotting
- Basic and enhanced plotting functions
- Training dynamics visualization
- Distribution comparison plots

## File Descriptions

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~60 | Entry point for training |
| `src/__init__.py` | ~25 | Package API exports |
| `src/config/settings.py` | ~28 | Configuration defaults |
| `src/models/quantum_generator.py` | ~115 | Quantum generator model |
| `src/models/classical_discriminator.py` | ~45 | Discriminator model |
| `src/training/trainer.py` | ~160 | Training loop |
| `src/utils/data.py` | ~25 | Data utilities |
| `src/utils/metrics.py` | ~30 | Metric computations |
| `src/visualization/plotting.py` | ~200 | Visualization functions |

## Benefits of This Structure

✅ **Modularity**: Each component is in its own file  
✅ **Maintainability**: Easy to find and update code  
✅ **Scalability**: Simple to add new features  
✅ **Reusability**: Import only what you need  
✅ **Professional**: Industry-standard structure  
✅ **Testable**: Easy to write unit tests

## Quick Navigation

- Need to change hyperparameters? → `src/config/settings.py`
- Want to modify the quantum circuit? → `src/models/quantum_generator.py`
- Need to adjust training logic? → `src/training/trainer.py`
- Want different data sampling? → `src/utils/data.py`
- Need new visualizations? → `src/visualization/plotting.py`

---

