"""
Quantum Generator model using Variational Quantum Circuit (VQC).
"""
import torch
import torch.nn as nn
import pennylane as qml
from typing import Dict, Any


def create_quantum_device(n_qubits: int):
    """Create a PennyLane quantum device."""
    return qml.device("default.qubit", wires=n_qubits)


def create_generator_qnode(device, n_qubits: int):
   
    @qml.qnode(device, interface="torch", diff_method="backprop")
    def generator_qnode(latent_batch: torch.Tensor, weights: torch.Tensor):
       
       
        for wire in range(n_qubits):
            qml.RY(latent_batch[:, wire], wires=wire)
            qml.RZ(latent_batch[:, wire], wires=wire)
      
       
        for layer_idx in range(weights.shape[0]):
            
            for wire in range(n_qubits):
                qml.RX(weights[layer_idx, wire, 0], wires=wire)
                qml.RY(weights[layer_idx, wire, 1], wires=wire)
                qml.RZ(weights[layer_idx, wire, 2], wires=wire)
            
            # Entanglement (circular CNOT ladder)
            if n_qubits > 1:
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])  # Circular
        
        # Measure expectation values
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]
    
    return generator_qnode


class QuantumGenerator(nn.Module):
   
    def __init__(self, config: Dict[str, Any], qnode=None):
       
        super().__init__()
        self.n_layers = config["n_layers"]
        self.n_qubits = config["n_qubits"]
        self.scale = config["scale_factor"]
        
        # Trainable quantum circuit parameters
        weight_shape = (self.n_layers, self.n_qubits, 3)
        self.theta = nn.Parameter(0.02 * torch.randn(weight_shape))
       
        if qnode is None:
            device = create_quantum_device(self.n_qubits)
            self.qnode = create_generator_qnode(device, self.n_qubits)
        else:
            self.qnode = qnode
    
    def forward(self, z_batch: torch.Tensor) -> torch.Tensor:
       
        expvals = self.qnode(z_batch, self.theta)
        return (self.scale * torch.stack(expvals, dim=1)).float()
