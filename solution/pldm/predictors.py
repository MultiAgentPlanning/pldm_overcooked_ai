"""
Module containing different types of predictor models for PLDM.

Each predictor takes state embeddings (and actions) as input and predicts
either the next state (dynamics) or expected reward (reward).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any

class BasePredictor(nn.Module):
    """Base class for all predictor models."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        """Forward pass of the predictor - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    @staticmethod
    def create(predictor_type: str, **kwargs):
        """
        Factory method to create a predictor of the given type.
        
        Args:
            predictor_type: Type of predictor to create ("grid", "transformer", or "lstm")
            **kwargs: Additional arguments to pass to the predictor constructor
            
        Returns:
            An instance of the requested predictor type
        """
        if predictor_type.lower() == "grid":
            return GridPredictor(**kwargs)
        elif predictor_type.lower() == "transformer":
            return TransformerPredictor(**kwargs)
        elif predictor_type.lower() == "lstm":
            return LSTMPredictor(**kwargs)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")


class GridPredictor(BasePredictor):
    """
    Convolutional predictor for grid-based state representations.
    
    This predictor uses convolutional layers to process grid states
    and fully connected layers to integrate actions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_actions: int = 6,
        action_embed_dim: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        channels: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        is_reward: bool = False,
    ):
        """
        Initialize the GridPredictor.
        
        Args:
            input_dim: Dimensionality of the input state embedding
            output_dim: Dimensionality of the output prediction
            num_actions: Number of possible actions
            action_embed_dim: Dimensionality of action embeddings
            hidden_size: Size of hidden layers
            num_layers: Number of hidden layers
            dropout: Dropout probability
            activation: Activation function to use
            channels: Number of input channels (for grid state)
            height: Height of the grid (for grid state)
            width: Width of the grid (for grid state)
            is_reward: Whether this predictor is for reward prediction
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim
        self.hidden_size = hidden_size
        self.is_reward = is_reward
        
        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        
        # For grid-based input
        if channels is not None and height is not None and width is not None:
            self.is_grid_input = True
            self.channels = channels
            self.height = height
            self.width = width
            
            # Convolutional layers
            self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            
            # Calculate flattened size after convolutions
            conv_output_size = 64 * height * width
            
            # Projection to hidden size
            self.proj = nn.Linear(conv_output_size, input_dim)
        else:
            self.is_grid_input = False
        
        # Get activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = F.relu
        
        # FC layers
        fc_layers = []
        
        # Input size is state embedding + action embedding for both agents
        total_input_dim = input_dim + 2 * action_embed_dim
        
        # First layer
        fc_layers.append(nn.Linear(total_input_dim, hidden_size))
        fc_layers.append(nn.Dropout(dropout))
        
        # Middle layers
        for _ in range(num_layers - 1):
            fc_layers.append(nn.Linear(hidden_size, hidden_size))
            fc_layers.append(nn.Dropout(dropout))
        
        self.fc_layers = nn.ModuleList(fc_layers)
        
        # Output projection
        if is_reward:
            # For reward prediction, output a single scalar
            self.output_proj = nn.Linear(hidden_size, 1)
        else:
            # For dynamics prediction, output the next state
            self.output_proj = nn.Linear(hidden_size, output_dim)
    
    def forward(self, state, action_indices, return_embedding=False):
        """
        Forward pass to predict the next state or reward.
        
        Args:
            state: The current state representation
            action_indices: Tensor of shape [batch_size, 2] with action indices for both agents
            return_embedding: If True, returns the state embedding
            
        Returns:
            Predicted next state or reward
        """
        batch_size = state.shape[0]
        
        # Process state
        if self.is_grid_input:
            # For grid-based input
            x = self.activation(self.conv1(state))
            x = self.activation(self.conv2(x))
            x = x.view(batch_size, -1)  # Flatten
            state_embed = self.proj(x)
        else:
            # Input is already an embedding
            state_embed = state
        
        # If only the embedding is requested, return it
        if return_embedding:
            return state_embed
        
        # Process actions
        action_embeds = []
        for i in range(action_indices.shape[1]):  # For each agent
            action_embed = self.action_embedding(action_indices[:, i])
            action_embeds.append(action_embed)
        
        # Concatenate all action embeddings
        action_embed = torch.cat(action_embeds, dim=1)
        
        # Concatenate state and action embeddings
        combined = torch.cat([state_embed, action_embed], dim=1)
        
        # Pass through FC layers
        x = combined
        for i in range(0, len(self.fc_layers), 2):
            x = self.activation(self.fc_layers[i](x))
            x = self.fc_layers[i+1](x)  # Apply dropout
        
        # Output projection
        if self.is_reward:
            # Return scalar reward
            return self.output_proj(x)
        else:
            # For dynamics, could be grid state or embedding
            return self.output_proj(x)


class TransformerPredictor(BasePredictor):
    """
    Transformer-based predictor for sequential state representations.
    
    This predictor uses transformer layers to process states and actions.
    Supports teacher forcing during training for improved stability.
    Compatible with both grid-based and vector-based state encoders.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_actions: int = 6,
        action_embed_dim: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        is_reward: bool = False,
        teacher_forcing_ratio: float = 0.5,
    ):
        """
        Initialize the TransformerPredictor.
        
        Args:
            input_dim: Dimensionality of the input state embedding
            output_dim: Dimensionality of the output prediction
            num_actions: Number of possible actions
            action_embed_dim: Dimensionality of action embeddings
            hidden_size: Size of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            is_reward: Whether this predictor is for reward prediction
            teacher_forcing_ratio: Probability of using teacher forcing during training (0.0 to 1.0)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim
        self.hidden_size = hidden_size
        self.is_reward = is_reward
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        
        # Project input to transformer dimension
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # Separate projection for action embeddings
        self.action_proj = nn.Linear(action_embed_dim, hidden_size)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        # Output projection
        if is_reward:
            # For reward prediction, output a single scalar
            self.output_proj = nn.Linear(hidden_size, 1)
        else:
            # For dynamics prediction, output the next state
            self.output_proj = nn.Linear(hidden_size, output_dim)
    
    def forward(self, state, action_indices, return_embedding=False, target_state=None, use_teacher_forcing=None):
        """
        Forward pass to predict the next state or reward.
        
        Args:
            state: The current state representation - can be either:
                  - Grid state: [batch_size, channels, height, width]
                  - Vector state: [batch_size, input_dim]
                  - Embedding from CNN: [batch_size, state_embed_dim]
            action_indices: Tensor of shape [batch_size, 2] with action indices for both agents
            return_embedding: If True, returns the state embedding instead of grid reconstruction
            target_state: Optional ground truth next state for teacher forcing
            use_teacher_forcing: Whether to use teacher forcing (if None, uses self.teacher_forcing_ratio)
            
        Returns:
            Predicted next state or reward
        """
        batch_size = state.shape[0]
        
        # Handle different input state formats
        is_grid_input = len(state.shape) > 2
        
        # Store original dimensions for reshaping the output later if grid input
        if is_grid_input:
            self.original_shape = state.shape
            # Flatten the grid dimensions
            state = state.view(batch_size, -1)
            
            # Also flatten target_state if provided and has more than 2 dimensions
            if target_state is not None and len(target_state.shape) > 2:
                target_state = target_state.view(batch_size, -1)
        
        # Project state to transformer dimension
        state_embed = self.input_proj(state)
        
        # Process actions
        action_embeds = []
        for i in range(action_indices.shape[1]):  # For each agent
            action_embed = self.action_embedding(action_indices[:, i])
            # Project to transformer dimension using action projection
            action_embed = self.action_proj(action_embed)
            action_embeds.append(action_embed)
        
        # Determine if we should use teacher forcing
        if use_teacher_forcing is None:
            use_teacher_forcing = target_state is not None and torch.rand(1).item() < self.teacher_forcing_ratio
        
        # Apply teacher forcing if enabled and target_state is provided
        if use_teacher_forcing and target_state is not None and self.training:
            # Project target state to transformer dimension
            target_embed = self.input_proj(target_state)
            
            # Stack state, target state, and actions as sequence
            # [batch_size, seq_len=4, hidden_size]
            sequence = torch.stack([state_embed, target_embed] + action_embeds, dim=1)
        else:
            # Stack state and actions as sequence [batch_size, seq_len=3, hidden_size]
            # where seq_len = 1 (state) + 2 (actions for 2 agents)
            sequence = torch.stack([state_embed] + action_embeds, dim=1)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(sequence)
        
        # Take the first output (corresponding to state) for the final prediction
        final_output = transformer_output[:, 0, :]
        
        # Output projection
        if self.is_reward:
            # Return scalar reward
            return self.output_proj(final_output)
        else:
            # For dynamics, get the output
            output = self.output_proj(final_output)
            
            # Return embedding directly if requested
            if return_embedding:
                return output
            
            # Otherwise reshape back to grid form if the input was a grid
            if is_grid_input and not return_embedding:
                output = output.view(batch_size, *self.original_shape[1:])
                
            return output


class LSTMPredictor(BasePredictor):
    """
    LSTM-based predictor for sequential state representations.
    
    This predictor uses LSTM layers to process states and actions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_actions: int = 6,
        action_embed_dim: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        is_reward: bool = False,
    ):
        """
        Initialize the LSTMPredictor.
        
        Args:
            input_dim: Dimensionality of the input state embedding
            output_dim: Dimensionality of the output prediction
            num_actions: Number of possible actions
            action_embed_dim: Dimensionality of action embeddings
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            is_reward: Whether this predictor is for reward prediction
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.is_reward = is_reward
        
        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        
        # Combined input size: state + actions
        combined_input_dim = input_dim + 2 * action_embed_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=combined_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection
        lstm_output_dim = hidden_size * self.num_directions
        if is_reward:
            # For reward prediction, output a single scalar
            self.output_proj = nn.Linear(lstm_output_dim, 1)
        else:
            # For dynamics prediction, output the next state
            self.output_proj = nn.Linear(lstm_output_dim, output_dim)
    
    def forward(self, state, action_indices, return_embedding=False):
        """
        Forward pass to predict the next state or reward.
        
        Args:
            state: The current state representation
            action_indices: Tensor of shape [batch_size, 2] with action indices for both agents
            return_embedding: If True, returns the state embedding
            
        Returns:
            Predicted next state or reward
        """
        batch_size = state.shape[0]
        
        # If only the embedding is requested, return it
        if return_embedding:
            return state
        
        # Process actions
        action_embeds = []
        for i in range(action_indices.shape[1]):  # For each agent
            action_embed = self.action_embedding(action_indices[:, i])
            action_embeds.append(action_embed)
        
        # Concatenate state and actions
        combined = torch.cat([state] + action_embeds, dim=1)
        
        # Reshape for LSTM: [batch_size, seq_len=1, input_size]
        lstm_input = combined.unsqueeze(1)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, 
                         device=state.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size,
                         device=state.device)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        
        # Get the output for the last time step
        if self.bidirectional:
            # For bidirectional, concatenate forward and backward outputs
            final_output = lstm_out[:, -1, :]
        else:
            final_output = lstm_out[:, -1, :]
        
        # Output projection
        if self.is_reward:
            # Return scalar reward
            return self.output_proj(final_output)
        else:
            # For dynamics, return embedding
            return self.output_proj(final_output)


def create_dynamics_predictor(config: Dict[str, Any], input_dim: int, output_dim: int):
    """
    Create a dynamics predictor based on configuration.
    
    Args:
        config: Configuration dictionary with predictor settings
        input_dim: Input dimension for the predictor
        output_dim: Output dimension for the predictor
        
    Returns:
        Initialized dynamics predictor
    """
    predictor_type = config.get("predictor_type", "grid").lower()
    
    # Common parameters
    num_actions = config.get("num_actions", 6)
    action_embed_dim = config.get("action_embed_dim", 4)
    hidden_size = config.get("hidden_size", 128)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.1)
    activation = config.get("activation", "relu")
    is_reward = False
    
    # Check if we're using a CNN encoder
    use_cnn_encoder = config.get("use_cnn_encoder", False)
    
    if predictor_type == "transformer":
        # Transformer parameters
        num_heads = config.get("nhead", 4)
        teacher_forcing_ratio = config.get("teacher_forcing_ratio", 0.5)
        
        return TransformerPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            num_actions=num_actions,
            action_embed_dim=action_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            is_reward=is_reward,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
    elif predictor_type == "lstm":
        # LSTM parameters
        bidirectional = config.get("bidirectional", False)
        
        return LSTMPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            num_actions=num_actions,
            action_embed_dim=action_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            is_reward=is_reward
        )
    else:  # Default to grid predictor (or any basic predictor for CNN embedding)
        # When using CNN encoder, we use grid predictor but with the input from CNN embedding
        channels = None
        height = None
        width = None
        
        if not use_cnn_encoder:
            # Only need these parameters for raw grid input
            grid_dims = config.get("grid_dims", {})
            channels = config.get("channels", 32)
            height = grid_dims.get("H")
            width = grid_dims.get("W")
        
        return GridPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            num_actions=num_actions,
            action_embed_dim=action_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            channels=channels,
            height=height,
            width=width,
            is_reward=is_reward
        )


def create_reward_predictor(config: Dict[str, Any], input_dim: int):
    """
    Create a reward predictor based on configuration.
    
    Args:
        config: Configuration dictionary with predictor settings
        input_dim: Input dimension for the predictor
        
    Returns:
        Initialized reward predictor
    """
    # Always output a scalar for reward
    output_dim = 1
    
    predictor_type = config.get("predictor_type", "grid").lower()
    
    # Common parameters
    num_actions = config.get("num_actions", 6)
    action_embed_dim = config.get("action_embed_dim", 4)
    hidden_size = config.get("hidden_size", 64)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.1)
    activation = config.get("activation", "relu")
    is_reward = True
    
    # Check if we're using a CNN encoder
    use_cnn_encoder = config.get("use_cnn_encoder", False)
    
    if predictor_type == "transformer":
        # Transformer parameters
        num_heads = config.get("nhead", 4)
        teacher_forcing_ratio = config.get("teacher_forcing_ratio", 0.5)
        
        return TransformerPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            num_actions=num_actions,
            action_embed_dim=action_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            is_reward=is_reward,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
    elif predictor_type == "lstm":
        # LSTM parameters
        bidirectional = config.get("bidirectional", False)
        
        return LSTMPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            num_actions=num_actions,
            action_embed_dim=action_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            is_reward=is_reward
        )
    else:  # Default to grid predictor (or any basic predictor for CNN embedding)
        # When using CNN encoder, we use grid predictor but with the input from CNN embedding
        channels = None
        height = None
        width = None
        
        if not use_cnn_encoder:
            # Only need these parameters for raw grid input
            grid_dims = config.get("grid_dims", {})
            channels = config.get("channels", 32)
            height = grid_dims.get("H")
            width = grid_dims.get("W")
        
        return GridPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            num_actions=num_actions,
            action_embed_dim=action_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            channels=channels,
            height=height,
            width=width,
            is_reward=is_reward
        ) 