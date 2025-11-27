import torch
import warnings
warnings.filterwarnings("ignore")

from src.config import get_default_config
from src.training import train_qgan
from src.visualization import plot_results_basic, plot_results_enhanced

def main():
   
    CONFIG = get_default_config()
    
    # Ensure output directory exists
    import os
    os.makedirs("output", exist_ok=True)
    
    CONFIG["n_qubits"] = 2
    CONFIG["device"] = "cpu"
    CONFIG["num_epochs"] = 300
    
    print("=" * 60)
    print(" QUANTUM GAN TRAINING")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
   
    trained_generator, trained_discriminator, training_history = train_qgan(CONFIG)
    
    print("\n Generating visualizations...")
    plot_results_basic(trained_generator, training_history, CONFIG)
    plot_results_enhanced(trained_generator, training_history, CONFIG)
    
   
    checkpoint_path = "output/qgan_checkpoint.pth"
    torch.save({
        'generator_state_dict': trained_generator.state_dict(),
        'discriminator_state_dict': trained_discriminator.state_dict(),
        'config': CONFIG,
        'history': training_history,
    }, checkpoint_path)
    
    print(f"\n✅ Model checkpoint saved to '{checkpoint_path}'")
    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

    
