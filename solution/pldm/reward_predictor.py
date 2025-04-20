import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional

from .dynamics_predictor import ActionEmbedding


class GridRewardPredictor(nn.Module):
    """
    Predicts reward from grid-based state representation and joint action.
    """
    def __init__(self, 
                 state_embed_dim: int = 128, 
                 action_embed_dim: int = 4,
                 num_actions: int = 6,
                 hidden_dim: int = 64,
                 num_channels: int = 31,
                 grid_height: int = 5,
                 grid_width: int = 13):
        super().__init__()
        
        self.state_embed_dim = state_embed_dim
        self.action_embed_dim = action_embed_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Convolutional layers for state processing
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # The encoder output projection to state_embed_dim - set dynamically in forward pass
        self.encoder_proj = None
        
        # Action embedding
        self.action_embedding = ActionEmbedding(num_actions, action_embed_dim)
        
        # Reward prediction network
        total_input_dim = state_embed_dim + 2 * action_embed_dim
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output a single scalar reward
    
    def forward(self, state, action_indices):
        """
        Forward pass to predict reward.
        
        Args:
            state: Tensor of shape [batch_size, channels, height, width]
            action_indices: Tensor of shape [batch_size, 2] with action indices for both agents
            
        Returns:
            Tensor of shape [batch_size, 1] representing the predicted reward
        """
        batch_size = state.shape[0]
        
        # Process through convolutional layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        
        # Flatten the convolutional output
        flattened_size = x.size(1) * x.size(2) * x.size(3)
        x = x.view(batch_size, -1)
        
        # Create encoder projection layer if needed or if dimensions changed
        if self.encoder_proj is None or self.encoder_proj.in_features != flattened_size:
            self.encoder_proj = nn.Linear(flattened_size, self.state_embed_dim).to(state.device)
            print(f"Reward: Created new encoder projection layer with input size {flattened_size}")
            
        # Encode state to fixed dimension
        state_embed = F.relu(self.encoder_proj(x))
        
        # Encode actions
        action_embed = self.action_embedding(action_indices)
        
        # Concatenate state and action embeddings
        combined = torch.cat([state_embed, action_embed], dim=1)
        
        # Adjust input size of first FC layer if needed
        combined_size = combined.size(1)
        expected_size = self.fc1.in_features
        
        if combined_size != expected_size:
            print(f"Reward: Recreating FC layers. Expected size: {expected_size}, got: {combined_size}")
            self.fc1 = nn.Linear(combined_size, self.hidden_dim).to(state.device)
        
        # Process through fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        reward = self.fc3(x)
        
        return reward


class VectorRewardPredictor(nn.Module):
    """
    Predicts reward from vector-based state representation and joint action.
    """
    def __init__(self, 
                 state_dim: int, 
                 action_embed_dim: int = 4,
                 num_actions: int = 6,
                 hidden_dim: int = 64):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_embed_dim = action_embed_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Action embedding
        self.action_embedding = ActionEmbedding(num_actions, action_embed_dim)
        
        # Total input dimension (state + joint action)
        total_input_dim = state_dim + 2 * action_embed_dim
        
        # Reward prediction network
        self.reward_net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single scalar reward
        )
    
    def forward(self, state_vec, action_indices):
        """
        Forward pass to predict reward.
        
        Args:
            state_vec: Tensor of shape [batch_size, state_dim]
            action_indices: Tensor of shape [batch_size, 2] with action indices for both agents
            
        Returns:
            Tensor of shape [batch_size, 1] representing the predicted reward
        """
        # Encode actions
        action_embed = self.action_embedding(action_indices)
        
        # Concatenate state and action embeddings
        combined = torch.cat([state_vec, action_embed], dim=1)
        
        # Predict reward
        reward = self.reward_net(combined)
        
        return reward


class SharedEncoderRewardPredictor(nn.Module):
    """
    Reward predictor that shares a state encoder with other components.
    Takes embedded states rather than raw states as input.
    """
    def __init__(self, 
                 state_embed_dim: int = 128, 
                 action_embed_dim: int = 4,
                 num_actions: int = 6,
                 hidden_dim: int = 64):
        super().__init__()
        
        self.state_embed_dim = state_embed_dim
        self.action_embed_dim = action_embed_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Action embedding
        self.action_embedding = ActionEmbedding(num_actions, action_embed_dim)
        
        # Total input dimension (state + joint action)
        total_input_dim = state_embed_dim + 2 * action_embed_dim
        
        # Reward prediction network
        self.reward_net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single scalar reward
        )
    
    def forward(self, state_embed, action_indices):
        """
        Forward pass to predict reward.
        
        Args:
            state_embed: Tensor of shape [batch_size, state_embed_dim]
            action_indices: Tensor of shape [batch_size, 2] with action indices for both agents
            
        Returns:
            Tensor of shape [batch_size, 1] representing the predicted reward
        """
        # Encode actions
        action_embed = self.action_embedding(action_indices)
        
        # Concatenate state and action embeddings
        combined = torch.cat([state_embed, action_embed], dim=1)
        
        # Predict reward
        reward = self.reward_net(combined)
        
        return reward 