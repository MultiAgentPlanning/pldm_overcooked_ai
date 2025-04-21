import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional

from .utils import ACTION_INDICES


class ActionEmbedding(nn.Module):
    """
    Embeds discrete actions into continuous vectors.
    """
    def __init__(self, num_actions: int = 6, action_embed_dim: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(num_actions, action_embed_dim)
    
    def forward(self, action_indices):
        """
        Forward pass to embed action indices.
        
        Args:
            action_indices: Integer tensor of shape [batch_size, 2] with action indices for both agents
            
        Returns:
            Tensor of shape [batch_size, 2 * action_embed_dim]
        """
        # Embed each agent's action
        agent1_embedding = self.embedding(action_indices[:, 0])
        agent2_embedding = self.embedding(action_indices[:, 1])
        
        # Concatenate embeddings
        joint_embedding = torch.cat([agent1_embedding, agent2_embedding], dim=1)
        
        return joint_embedding


class GridDynamicsPredictor(nn.Module):
    """
    Predicts the next grid state given the current grid state and joint action.
    """
    def __init__(self, 
                 state_embed_dim: int = 128, 
                 action_embed_dim: int = 4,
                 num_actions: int = 6,
                 hidden_dim: int = 256,
                 num_channels: int = 31,
                 grid_height: int = 5,
                 grid_width: int = 13):
        super().__init__()
        
        self.state_embed_dim = state_embed_dim
        self.action_embed_dim = action_embed_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        # State encoder (CNN) - use separate conv layers instead of Sequential for dynamic handling
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Action embedding
        self.action_embedding = ActionEmbedding(num_actions, action_embed_dim)
        
        # Fully connected layers - now dynamic based on flattened size
        self.fc1 = nn.Linear(state_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # The encoder output projection to state_embed_dim
        self.encoder_proj = None  # Will be set dynamically in forward pass
        
        # Output layer set dynamically in forward pass
        self.output_proj = None
        
    def forward(self, state, action_indices, return_embedding=False):
        """
        Forward pass to predict the next state.
        
        Args:
            state: Tensor of shape [batch_size, channels, height, width]
            action_indices: Tensor of shape [batch_size, 2] with action indices for both agents
            return_embedding: If True, returns the internal state embedding instead of the next state prediction
            
        Returns:
            If return_embedding is False: Tensor of shape [batch_size, channels, height, width] representing the predicted next state
            If return_embedding is True: Tensor of shape [batch_size, state_embed_dim] representing the state embedding
        """
        batch_size, num_channels, grid_height, grid_width = state.shape
        
        # Process through convolutional layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the convolutional output
        flattened_size = x.size(1) * x.size(2) * x.size(3)
        x = x.view(batch_size, -1)
        
        # Create encoder projection layer if needed or if dimensions changed
        if self.encoder_proj is None or self.encoder_proj.in_features != flattened_size:
            self.encoder_proj = nn.Linear(flattened_size, self.state_embed_dim).to(state.device)
            print(f"Created new encoder projection layer with input size {flattened_size}")
            
        # Encode state to fixed dimension
        state_embed = F.relu(self.encoder_proj(x))
        
        # If only the embedding is requested, return it here
        if return_embedding:
            return state_embed
        
        # Encode actions
        action_embed = self.action_embedding(action_indices)
        
        # Concatenate state and action embeddings
        combined = torch.cat([state_embed, action_embed], dim=1)
        
        # Adjust input size of first FC layer if needed
        combined_size = combined.size(1)
        expected_size = self.fc1.in_features
        
        if combined_size != expected_size:
            print(f"Recreating FC layers. Expected size: {expected_size}, got: {combined_size}")
            self.fc1 = nn.Linear(combined_size, self.hidden_dim).to(state.device)
        
        # Process through fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        
        # Create output projection layer if needed or if dimensions changed
        output_size = num_channels * grid_height * grid_width
        if self.output_proj is None or self.output_proj.out_features != output_size:
            self.output_proj = nn.Linear(self.hidden_dim, output_size).to(state.device)
            print(f"Created new output projection layer with output size {output_size}")
            
        # Get flattened next state prediction
        next_state_flat = self.output_proj(x)
        
        # Reshape to grid form
        next_state = next_state_flat.view(batch_size, num_channels, grid_height, grid_width)
        
        return next_state


class VectorDynamicsPredictor(nn.Module):
    """
    Predicts the next state vector given the current state vector and joint action.
    """
    def __init__(self, 
                 state_dim: int, 
                 action_embed_dim: int = 4,
                 num_actions: int = 6,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_embed_dim = action_embed_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Action embedding
        self.action_embedding = ActionEmbedding(num_actions, action_embed_dim)
        
        # Total input dimension (state + joint action embedding)
        total_input_dim = state_dim + 2 * action_embed_dim
        
        # Dynamics prediction network
        self.dynamics_net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state_vec, action_indices, return_embedding=False):
        """
        Forward pass to predict the next state.
        
        Args:
            state_vec: Tensor of shape [batch_size, state_dim]
            action_indices: Tensor of shape [batch_size, 2] with action indices for both agents
            return_embedding: If True, returns the state embedding instead of the next state
            
        Returns:
            If return_embedding is False: Tensor of shape [batch_size, state_dim] representing the predicted next state
            If return_embedding is True: Tensor of shape [batch_size, state_dim] representing the state embedding
        """
        # If only the state vector is requested as embedding, return it directly
        if return_embedding:
            return state_vec
            
        # Encode actions
        action_embed = self.action_embedding(action_indices)
        
        # Concatenate state and action embeddings
        combined = torch.cat([state_vec, action_embed], dim=1)
        
        # Predict next state
        next_state = self.dynamics_net(combined)
        
        return next_state


class SharedEncoderDynamicsPredictor(nn.Module):
    """
    Dynamics predictor that shares a state encoder with other components.
    Takes embedded states rather than raw states as input.
    """
    def __init__(self, 
                 state_embed_dim: int = 128, 
                 action_embed_dim: int = 4,
                 num_actions: int = 6,
                 hidden_dim: int = 128,
                 output_dim: int = None):
        super().__init__()
        
        self.state_embed_dim = state_embed_dim
        self.action_embed_dim = action_embed_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else state_embed_dim
        
        # Action embedding
        self.action_embedding = ActionEmbedding(num_actions, action_embed_dim)
        
        # Total embedding dimension (state + joint action)
        total_embed_dim = state_embed_dim + 2 * action_embed_dim
        
        # Dynamics prediction network
        self.dynamics_net = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
    
    def forward(self, state_embed, action_indices):
        """
        Forward pass to predict the next state embedding.
        
        Args:
            state_embed: Tensor of shape [batch_size, state_embed_dim]
            action_indices: Tensor of shape [batch_size, 2] with action indices for both agents
            
        Returns:
            Tensor of shape [batch_size, output_dim] representing the predicted next state embedding
        """
        # Encode actions
        action_embed = self.action_embedding(action_indices)
        
        # Concatenate state and action embeddings
        combined = torch.cat([state_embed, action_embed], dim=1)
        
        # Predict next state embedding
        next_state_embed = self.dynamics_net(combined)
        
        return next_state_embed 