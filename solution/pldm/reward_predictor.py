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
        
        # State encoder (CNN) - similar to dynamics predictor
        self.state_encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_height * grid_width, 256),
            nn.ReLU(),
            nn.Linear(256, state_embed_dim)
        )
        
        # Action embedding
        self.action_embedding = ActionEmbedding(num_actions, action_embed_dim)
        
        # Total input dimension (state + joint action)
        total_input_dim = state_embed_dim + 2 * action_embed_dim
        
        # Reward prediction network - smaller than dynamics as it's a simpler task
        self.reward_net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single scalar reward
        )
    
    def forward(self, state, action_indices):
        """
        Forward pass to predict reward.
        
        Args:
            state: Tensor of shape [batch_size, channels, height, width]
            action_indices: Tensor of shape [batch_size, 2] with action indices for both agents
            
        Returns:
            Tensor of shape [batch_size, 1] representing the predicted reward
        """
        # Encode state
        state_embed = self.state_encoder(state)
        
        # Encode actions
        action_embed = self.action_embedding(action_indices)
        
        # Concatenate state and action embeddings
        combined = torch.cat([state_embed, action_embed], dim=1)
        
        # Predict reward
        reward = self.reward_net(combined)
        
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