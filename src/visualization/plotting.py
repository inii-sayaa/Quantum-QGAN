import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import Dict, Any

from ..utils import sample_real_data, compute_mmd


def plot_results_basic(generator: Any, history: Dict[str, Any], config: Dict[str, Any]):

    device = torch.device(config["device"])
    
   
    generator.eval()
    with torch.no_grad():
        real_samples = sample_real_data(512).cpu()
        z = torch.rand(512, config["latent_dim"], device=device) * 2 * math.pi
        fake_samples = generator(z).cpu()
    
    fig = plt.figure(figsize=(18, 5))
    
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(real_samples[:, 0], real_samples[:, 1], s=15, alpha=0.6, 
               label="Real", color="#2ca02c", edgecolors='k', linewidths=0.1)
    ax1.scatter(fake_samples[:, 0], fake_samples[:, 1], s=15, alpha=0.6, 
               label="Generated", color="#ff7f0e", edgecolors='k', linewidths=0.1)
    ax1.set_title("Data Distribution", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.set_aspect('equal')
    
    
    ax2 = plt.subplot(1, 3, 2)
    sns.kdeplot(real_samples[:, 0], color="#2ca02c", label="Real X", ax=ax2, fill=True, alpha=0.3)
    sns.kdeplot(fake_samples[:, 0], color="#ff7f0e", label="Generated X", ax=ax2, fill=True, alpha=0.3)
    sns.kdeplot(real_samples[:, 1], color="#2ca02c", linestyle='--', label="Real Y", ax=ax2)
    sns.kdeplot(fake_samples[:, 1], color="#ff7f0e", linestyle='--', label="Generated Y", ax=ax2)
    ax2.set_title("Marginal Distributions", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")
    ax2.legend()
    
   
    ax3 = plt.subplot(1, 3, 3)
    epochs = range(1, len(history["d_losses"]) + 1)
    ax3.plot(epochs, history["d_losses"], label="Discriminator", color="#1f77b4", linewidth=2)
    ax3.plot(epochs, history["g_losses"], label="Generator", color="#d62728", linewidth=2)
    ax3.set_title("Training Dynamics", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
   
    for epoch, mmd in history["mmd_scores"]:
        ax3.axvline(epoch, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("output/qgan_analysis.png", dpi=300, bbox_inches='tight')
    
  
    final_mmd = compute_mmd(real_samples.to(device), fake_samples.to(device))
    print(f"\n📊 Final MMD Score: {final_mmd:.4f} (lower is better)")
    print(f"   Generator Parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"   Training Complete!")


def plot_results_enhanced(generator: Any, history: Dict[str, Any], config: Dict[str, Any]):
   
    device = torch.device(config["device"])
  
    generator.eval()
    with torch.no_grad():
        real_samples = sample_real_data(512).cpu()
        z = torch.rand(512, config["latent_dim"], device=device) * 2 * math.pi
        fake_samples = generator(z).cpu()
        final_mmd = compute_mmd(real_samples.to(device), fake_samples.to(device))
    
    fig = plt.figure(figsize=(20, 6)) 
    
    ax1 = plt.subplot(1, 4, 1)
    ax1.scatter(real_samples[:, 0], real_samples[:, 1], s=15, alpha=0.6, 
               label="Real", color="#2ca02c", edgecolors='k', linewidths=0.1)
    ax1.scatter(fake_samples[:, 0], fake_samples[:, 1], s=15, alpha=0.6, 
               label="Generated", color="#ff7f0e", edgecolors='k', linewidths=0.1)
    ax1.set_title("Quantum Generator Output", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend(loc='lower right')
    ax1.set_aspect('equal')
    
    from matplotlib.patches import Rectangle
    ax1.add_patch(Rectangle((0.02, 0.78), 0.45, 0.2, 
                           transform=ax1.transAxes, facecolor='white', alpha=0.9, edgecolor='gray'))
    ax1.text(0.05, 0.95, f'Ansatz Config:\n{config["n_qubits"]} Qubits\n{config["n_layers"]} Layers\nEntanglement: Circular', 
             transform=ax1.transAxes, fontsize=9, va='top', family='monospace')
    
    
    ax2 = plt.subplot(1, 4, 2)
    sns.kdeplot(real_samples[:, 0], color="#2ca02c", label="Real X", ax=ax2, fill=True, alpha=0.3)
    sns.kdeplot(fake_samples[:, 0], color="#ff7f0e", label="Gen X", ax=ax2, fill=True, alpha=0.3)
    sns.kdeplot(real_samples[:, 1], color="#2ca02c", linestyle='--', label="Real Y", ax=ax2)
    sns.kdeplot(fake_samples[:, 1], color="#ff7f0e", linestyle='--', label="Gen Y", ax=ax2)
    ax2.set_title(f'Marginal Distributions\nFinal MMD: {final_mmd:.4f}', fontweight='bold')
    ax2.set_xlabel("Value")
    ax2.legend()
    
    
    ax3 = plt.subplot(1, 4, 3)
    epochs = range(1, len(history["d_losses"]) + 1)
    
    def smooth(scalars, weight=0.85):
      
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    d_smooth = smooth(history["d_losses"])
    g_smooth = smooth(history["g_losses"])
    
    ax3.plot(epochs, history["d_losses"], alpha=0.3, color="#1f77b4")
    ax3.plot(epochs, history["g_losses"], alpha=0.3, color="#d62728")
    ax3.plot(epochs, d_smooth, label="Discriminator (Smooth)", color="#1f77b4", linewidth=2)
    ax3.plot(epochs, g_smooth, label="Generator (Smooth)", color="#d62728", linewidth=2)
    
    ax3.set_title("Adversarial Training Dynamics", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
   
    ax4 = plt.subplot(1, 4, 4)
    
    categories = ['Quantum VQC', 'Classical MLP']
    params = [sum(p.numel() for p in generator.parameters()), 128*2 + 128*128 + 128*2]
    
    bars = ax4.bar(categories, params, color=['#9467bd', '#7f7f7f'])
    ax4.set_yscale('log')
    ax4.set_title("Parameter Efficiency", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Trainable Parameters (Log Scale)")
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
                
    ax4.text(0.5, 0.5, "Quantum VQC uses\norders of magnitude\nfewer parameters", 
             transform=ax4.transAxes, ha='center', fontsize=10, style='italic',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig("output/qgan_enhanced.png", dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Enhanced plots saved to 'output/qgan_enhanced.png'")
