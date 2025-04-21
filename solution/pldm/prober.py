import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Optional

class Prober(nn.Module):
    """
    A neural network that probes the latent representations of the PLDM model.
    Can be used to predict features like state components or rewards from the embeddings.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        activation: Optional[str] = "relu",
        arch: str = "mlp"
    ):
        """
        Initialize a prober network.
        
        Args:
            input_dim: Dimension of the input (embedding dimension)
            output_dim: Dimension of the output (e.g., 2 for 2D position)
            hidden_dims: Dimensions of hidden layers
            activation: Activation function to use ('relu', 'tanh', etc.)
            arch: Architecture type ('mlp', 'conv', etc.)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        
        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()  # Default
        
        # Build network architecture
        if arch == "mlp":
            layers = []
            prev_dim = input_dim
            
            for dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(self.activation)
                prev_dim = dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)
        
        elif arch == "conv":
            # For grid-based representations, we might want a convolutional prober
            # This is simplified and would need to be adapted to the actual input shape
            self.network = nn.Sequential(
                nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128),  # Adjust size based on input dimensions
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
    
    def forward(self, x):
        """Forward pass through the prober network."""
        return self.network(x)


class GridProber(Prober):
    """
    A prober specifically designed for grid-based PLDM representations.
    Handles multi-channel grid inputs.
    """
    def __init__(
        self,
        input_channels: int,
        grid_height: int,
        grid_width: int,
        output_dim: int,
        hidden_channels: Tuple[int, ...] = (32, 64),
        fc_dims: Tuple[int, ...] = (128,)
    ):
        """
        Args:
            input_channels: Number of input channels in the grid representation
            grid_height: Height of the grid
            grid_width: Width of the grid
            output_dim: Dimension of the output (e.g., 2 for 2D position)
            hidden_channels: Channels in hidden convolutional layers
            fc_dims: Dimensions of fully connected layers after convolutions
        """
        super(Prober, self).__init__()  # Call nn.Module init directly
        
        self.input_channels = input_channels
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.output_dim = output_dim
        
        # Convolutional layers
        conv_layers = []
        prev_channels = input_channels
        
        for channels in hidden_channels:
            conv_layers.append(nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            prev_channels = channels
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        # Calculate the flattened size after convolutions
        # (Assuming no pooling, so dimensions stay the same)
        flattened_size = prev_channels * grid_height * grid_width
        
        # Fully connected layers
        fc_layers = []
        prev_dim = flattened_size
        
        for dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            prev_dim = dim
            
        fc_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.fc_net = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        """
        Forward pass through the GridProber.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
        
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Apply convolutional layers
        x = self.conv_net(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        return self.fc_net(x)


class VectorProber(Prober):
    """
    A prober specifically designed for vector-based PLDM representations.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64)
    ):
        """
        Initialize a vector prober.
        
        Args:
            input_dim: Dimension of the input vector representation
            output_dim: Dimension of the output (e.g., 2 for 2D position)
            hidden_dims: Dimensions of hidden layers
        """
        # Use the standard MLP architecture from the parent class
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            arch="mlp"
        ) 