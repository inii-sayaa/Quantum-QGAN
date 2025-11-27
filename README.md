# 🌌 Quantum-QGAN: Hybrid Quantum-Classical Generative Adversarial Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-green.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **hybrid quantum-classical Generative Adversarial Network (GAN)** implementation that combines:
- 🔬 **Quantum Generator**: Variational Quantum Circuit (VQC) using PennyLane
- 🧠 **Classical Discriminator**: Multi-layer Perceptron (MLP) using PyTorch
- 📊 **Advanced Metrics**: Maximum Mean Discrepancy (MMD) for distribution comparison

This project demonstrates how quantum computing can be integrated into modern machine learning workflows for generative modeling tasks.

---

## 🚀 Features

- ✅ **Hybrid Architecture**: Quantum generator + Classical discriminator
- ✅ **Flexible Configuration**: Easy-to-modify hyperparameters
- ✅ **Professional Structure**: Modular, maintainable, and scalable codebase
- ✅ **Rich Visualizations**: Training dynamics and distribution analysis
- ✅ **Reproducibility**: Seed management for consistent results
- ✅ **Model Checkpointing**: Save and resume training

---

## 📁 Project Structure

```
QGAN-QUANTUM/
│
├── 📄 main.py                    # Main entry point
├── 📄 requirements.txt           # Dependencies
├── 📄 STRUCTURE.md              # Detailed structure guide
│
├── 📁 src/                       # Source code
│   ├── config.py                 # Configuration settings
│   ├── models.py                 # Quantum & Classical models
│   ├── training.py               # Training pipeline
│   ├── visualization.py          # Plotting functions
│   └── utils.py                  # Utility functions
│
├── 📁 data/                      # Data directory
├── 📁 output/                    # Results & checkpoints
│   ├── qgan_analysis.png         # Training plots
│   ├── qgan_enhanced.png         # Enhanced visualization
│   └── qgan_checkpoint.pth       # Model checkpoint
│
└── 📁 venv/                      # Virtual environment
```

See [STRUCTURE.md](STRUCTURE.md) for detailed architecture documentation.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/inii-sayaa/Quantum-QGAN.git
   cd Quantum-QGAN
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎯 Quick Start

### Basic Training

Run with default configuration:
```bash
python main.py
```

This will:
1. Initialize a 2-qubit quantum generator
2. Train for 300 epochs on CPU
3. Generate visualization plots
4. Save model checkpoint

### Configuration

Modify hyperparameters in `main.py`:

```python
CONFIG = get_default_config()

# Adjust settings
CONFIG["n_qubits"] = 3          # Number of qubits
CONFIG["num_epochs"] = 500      # Training epochs
CONFIG["batch_size"] = 64       # Batch size
CONFIG["lr_gen"] = 0.01         # Generator learning rate
CONFIG["lr_disc"] = 0.001       # Discriminator learning rate
CONFIG["device"] = "cuda"       # Use GPU (if available)
```

---

## 📊 Outputs

After training, you'll find:

### 1. **Training Analysis** (`output/qgan_analysis.png`)
- Loss curves (Generator & Discriminator)
- MMD metric over time
- Sample distribution comparison
- Training statistics

### 2. **Enhanced Visualization** (`output/qgan_enhanced.png`)
- Distribution heatmaps
- Quantum circuit diagram
- Convergence analysis
- Advanced metrics

### 3. **Model Checkpoint** (`output/qgan_checkpoint.pth`)
- Trained generator state
- Trained discriminator state
- Configuration snapshot
- Training history

---

## 🧪 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│          Quantum GAN Training Loop              │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. Noise Input (Latent Vector)                │
│           ↓                                     │
│  2. Quantum Generator (VQC)                    │
│           ↓                                     │
│  3. Generated Samples                          │
│           ↓                                     │
│  4. Classical Discriminator (MLP)              │
│           ↓                                     │
│  5. Real/Fake Classification                   │
│           ↓                                     │
│  6. Adversarial Loss & Backpropagation         │
│           ↓                                     │
│  7. Parameter Updates                          │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Quantum Generator
- **Variational Quantum Circuit (VQC)** with parameterized gates
- Encodes classical noise into quantum states
- Implements quantum rotations and entanglement
- Outputs generated samples via measurement

### Classical Discriminator
- **Multi-Layer Perceptron (MLP)** with batch normalization
- Distinguishes real data from generated samples
- Standard adversarial training objective

---

## 📈 Key Metrics

- **Generator Loss**: Measures generator's ability to fool discriminator
- **Discriminator Loss**: Measures discriminator's classification accuracy
- **MMD (Maximum Mean Discrepancy)**: Distribution similarity metric
- **Mean/Std Difference**: Statistical comparison of distributions

---

## 🔧 Advanced Usage

### Resuming Training

Load a previous checkpoint:
```python
checkpoint = torch.load("output/qgan_checkpoint.pth")
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
```

### Custom Data Distribution

Modify target distribution in `src/utils.py`:
```python
def sample_real_data(n_samples, config):
    # Your custom data distribution
    return custom_data
```

---

## 🎓 Learning Resources

### Quantum Computing
- [PennyLane Documentation](https://pennylane.ai/)
- [Quantum Machine Learning](https://pennylane.ai/qml/)

### GANs
- [GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Original GAN Paper](https://arxiv.org/abs/1406.2661)

### Hybrid Quantum-Classical ML
- [Quantum GANs](https://arxiv.org/abs/1804.08641)
- [Variational Quantum Algorithms](https://arxiv.org/abs/2012.09265)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **PennyLane**: Quantum computing framework
- **PyTorch**: Deep learning framework
- **Quantum Computing Community**: Research and inspiration

---

## 📧 Contact

**Author**: [Your Name]  
**GitHub**: [@inii-sayaa](https://github.com/inii-sayaa)  
**Project**: [Quantum-QGAN](https://github.com/inii-sayaa/Quantum-QGAN)

---

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐!

---

*Built with 💜 using Quantum Computing and Machine Learning*
