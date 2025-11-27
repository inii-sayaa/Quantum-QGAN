import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from typing import Dict, Any, Tuple
import numpy as np

from ..models import QuantumGenerator, ClassicalDiscriminator, create_quantum_device, create_generator_qnode
from ..utils import sample_real_data, compute_mmd


def set_global_seeds(seed: int):
   
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_qgan(config: Dict[str, Any]) -> Tuple[QuantumGenerator, ClassicalDiscriminator, Dict[str, Any]]:
   
    set_global_seeds(config["seed"])
    
    device = torch.device(config["device"])
    
   
    quantum_dev = create_quantum_device(config["n_qubits"])
    qnode = create_generator_qnode(quantum_dev, config["n_qubits"])
    
    
    generator = QuantumGenerator(config, qnode=qnode).to(device)
    discriminator = ClassicalDiscriminator().to(device)
    
  
    optimizer_G = optim.Adam(
        generator.parameters(), 
        lr=config["lr_G"], 
        weight_decay=config["weight_decay"]
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(), 
        lr=config["lr_D"], 
        weight_decay=config["weight_decay"]
    )
    
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=config["num_epochs"])
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=config["num_epochs"])
    
   
    criterion = nn.BCEWithLogitsLoss()
    
   
    real_dataset = sample_real_data(config["dataset_size"])
    dataloader = DataLoader(real_dataset, batch_size=config["batch_size"], shuffle=True)
    
   
    history = {
        "d_losses": [],
        "g_losses": [],
        "mmd_scores": [],
    }
    
    print("🚀 Starting qGAN Training...")
    print(f"   Quantum Device: {quantum_dev.name}")
    print(f"   Parameters: {sum(p.numel() for p in generator.parameters())} (quantum)")
    
    for epoch in range(1, config["num_epochs"] + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        
        for real_batch in dataloader:
            real_batch = real_batch.to(device)
            batch_size_curr = real_batch.size(0)
            
           
            real_labels = torch.ones((batch_size_curr, 1), device=device)
            fake_labels = torch.zeros((batch_size_curr, 1), device=device)
            
            
            optimizer_D.zero_grad()
            
            
            real_logits = discriminator(real_batch)
            loss_real = criterion(real_logits, real_labels)
            
           
            z = torch.rand(batch_size_curr, config["latent_dim"], device=device) * 2 * math.pi
            fake_batch = generator(z)
            fake_logits = discriminator(fake_batch.detach())
            loss_fake = criterion(fake_logits, fake_labels)
            
            loss_D = loss_real + loss_fake
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config["gradient_clip"])
            optimizer_D.step()
            
          
            optimizer_G.zero_grad()
            
            z = torch.rand(batch_size_curr, config["latent_dim"], device=device) * 2 * math.pi
            gen_samples = generator(z)
            gen_logits = discriminator(gen_samples)
            loss_G = criterion(gen_logits, real_labels)  
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), config["gradient_clip"])
            optimizer_G.step()
            
            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()
        
       
        scheduler_G.step()
        scheduler_D.step()
        
       
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        history["d_losses"].append(avg_d_loss)
        history["g_losses"].append(avg_g_loss)
        
       
        if epoch % 20 == 0 or epoch == 1:
            generator.eval()
            with torch.no_grad():

                real_val = sample_real_data(512).to(device)
                z_val = torch.rand(512, config["latent_dim"], device=device) * 2 * math.pi
                fake_val = generator(z_val)
                
                mmd_score = compute_mmd(real_val, fake_val)
                history["mmd_scores"].append((epoch, mmd_score))
                
                print(f"Epoch {epoch:03d} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | MMD: {mmd_score:.4f}")
            generator.train()
    
    return generator, discriminator, history
