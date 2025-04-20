import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union, Optional, Callable
import time
import json
from pathlib import Path
import torch.nn.functional as F

from .state_encoder import GridStateEncoder, VectorStateEncoder, StateEncoderNetwork, VectorEncoderNetwork
from .dynamics_predictor import GridDynamicsPredictor, VectorDynamicsPredictor, SharedEncoderDynamicsPredictor
from .reward_predictor import GridRewardPredictor, VectorRewardPredictor, SharedEncoderRewardPredictor
from .data_processor import get_overcooked_dataloaders


class PLDMTrainer:
    """
    Trainer for Predictive Latent Dynamics Model (PLDM) for Overcooked.
    Implements training procedures for the dynamics and reward models.
    """
    def __init__(
        self, 
        data_path=None,
        output_dir="checkpoints",
        model_type="grid",
        batch_size=32,
        lr=1e-3,
        state_embed_dim=128,
        action_embed_dim=4,
        num_actions=6,
        dynamics_hidden_dim=128,
        reward_hidden_dim=64,
        grid_height=None,
        grid_width=None,
        device=None,
        max_samples=None,
        num_workers=4,
        env=None,
        state_encoder=None,
        gamma=0.99,
    ):
        # Setup output directory
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store configuration parameters
        self.model_type = model_type
        self.batch_size = int(batch_size) if not isinstance(batch_size, int) else batch_size
        self.lr = float(lr) if not isinstance(lr, float) else lr
        self.state_embed_dim = int(state_embed_dim) if not isinstance(state_embed_dim, int) else state_embed_dim
        self.action_embed_dim = int(action_embed_dim) if not isinstance(action_embed_dim, int) else action_embed_dim
        self.num_actions = int(num_actions) if not isinstance(num_actions, int) else num_actions
        self.dynamics_hidden_dim = int(dynamics_hidden_dim) if not isinstance(dynamics_hidden_dim, int) else dynamics_hidden_dim
        self.reward_hidden_dim = int(reward_hidden_dim) if not isinstance(reward_hidden_dim, int) else reward_hidden_dim
        
        # Handle grid dimensions that might be None or strings
        if grid_height is not None and not isinstance(grid_height, int):
            self.grid_height = int(grid_height)
        else:
            self.grid_height = grid_height
            
        if grid_width is not None and not isinstance(grid_width, int):
            self.grid_width = int(grid_width)
        else:
            self.grid_width = grid_width
            
        # Handle max_samples that might be None or a string
        if max_samples is not None and not isinstance(max_samples, int):
            self.max_samples = int(max_samples)
        else:
            self.max_samples = max_samples
            
        # Handle num_workers that might be a string
        self.num_workers = int(num_workers) if not isinstance(num_workers, int) else num_workers
        
        # Make sure gamma is a float
        self.gamma = float(gamma) if not isinstance(gamma, float) else gamma
        
        # Print parameters for debugging
        print(f"Initializing PLDMTrainer with: lr={self.lr}, batch_size={self.batch_size}, model_type={self.model_type}")
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Initialize state encoder based on model type
        if state_encoder is not None:
            self.state_encoder = state_encoder
        elif model_type == "grid":
            self.state_encoder = GridStateEncoder(grid_height=grid_height, grid_width=grid_width)
        else:  # model_type == "vector"
            self.state_encoder = VectorStateEncoder(grid_height=grid_height, grid_width=grid_width)
        
        # Initialize models and optimizers
        self.dynamics_model = None
        self.reward_model = None
        self.dynamics_optimizer = None
        self.reward_optimizer = None
        self.dynamics_criterion = nn.MSELoss()
        self.reward_criterion = nn.MSELoss()
        
        # Load and prepare data if path is provided
        if data_path:
            self.train_loader, self.val_loader = get_overcooked_dataloaders(
                data_path=data_path,
                state_encoder_type=model_type,
                batch_size=batch_size,
                max_samples=max_samples,
                num_workers=num_workers
            )
            
            # Initialize models based on a sample from the dataset
            self._initialize_models()
        else:
            self.train_loader = None
            self.val_loader = None
        
        # Tracking metrics
        self.dynamics_losses = []
        self.reward_losses = []
        
    def _initialize_models(self):
        """Initialize the dynamics and reward models based on model type."""
        # Get a sample batch to determine dimensions
        sample_batch = next(iter(self.train_loader))
        state, action, _, _ = sample_batch
        
        # Get state shape
        if self.model_type == "grid":
            batch_size, num_channels, grid_height, grid_width = state.shape
            
            # Update grid dimensions if not explicitly set
            if self.grid_height is None:
                self.grid_height = grid_height
            if self.grid_width is None:
                self.grid_width = grid_width
                
            print(f"Initializing grid models with dimensions: channels={num_channels}, height={self.grid_height}, width={self.grid_width}")
            
            # Initialize grid-based models
            self.dynamics_model = GridDynamicsPredictor(
                state_embed_dim=self.state_embed_dim,
                action_embed_dim=self.action_embed_dim,
                num_actions=self.num_actions,
                hidden_dim=self.dynamics_hidden_dim,
                num_channels=num_channels,
                grid_height=self.grid_height,
                grid_width=self.grid_width
            ).to(self.device)
            
            self.reward_model = GridRewardPredictor(
                state_embed_dim=self.state_embed_dim,
                action_embed_dim=self.action_embed_dim,
                num_actions=self.num_actions,
                hidden_dim=self.reward_hidden_dim,
                num_channels=num_channels,
                grid_height=self.grid_height,
                grid_width=self.grid_width
            ).to(self.device)
        else:  # model_type == "vector"
            batch_size, state_dim = state.shape
            
            # Initialize vector-based models
            self.dynamics_model = VectorDynamicsPredictor(
                state_dim=state_dim,
                action_embed_dim=self.action_embed_dim,
                num_actions=self.num_actions,
                hidden_dim=self.dynamics_hidden_dim
            ).to(self.device)
            
            self.reward_model = VectorRewardPredictor(
                state_dim=state_dim,
                action_embed_dim=self.action_embed_dim,
                num_actions=self.num_actions,
                hidden_dim=self.reward_hidden_dim
            ).to(self.device)
        
        # Initialize optimizers
        self.dynamics_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=self.lr)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=self.lr)
        
        # Print the size of the model to verify
        total_dynamics_params = sum(p.numel() for p in self.dynamics_model.parameters() if p.requires_grad)
        total_reward_params = sum(p.numel() for p in self.reward_model.parameters() if p.requires_grad)
        
        print(f"Initialized {self.model_type} models with grid size {self.grid_height}x{self.grid_width}")
        print(f"Dynamics model: {self.dynamics_model}")
        print(f"Reward model: {self.reward_model}")
        print(f"Total trainable parameters - Dynamics: {total_dynamics_params:,}, Reward: {total_reward_params:,}")
    
    def train_episode(self, episode_data):
        """
        Train the dynamics and reward models on a single episode.
        
        Args:
            episode_data: List of (state, action, next_state, reward) tuples
            
        Returns:
            Tuple of (dynamics_loss, reward_loss)
        """
        if not episode_data:
            return 0.0, 0.0
        
        # Cumulative losses for this episode
        episode_dynamics_loss = 0.0
        episode_reward_loss = 0.0
        
        # Iterate over transitions in the episode
        for state, action, next_state, reward in episode_data:
            # Encode current and next states
            state_grid = self.state_encoder.encode(state)
            next_state_grid = self.state_encoder.encode(next_state)
            
            # Convert to PyTorch tensors
            state_tensor = torch.FloatTensor(state_grid).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state_grid).unsqueeze(0).to(self.device)
            reward_tensor = torch.FloatTensor([reward]).to(self.device)
            
            # Train dynamics model
            self.dynamics_optimizer.zero_grad()
            predicted_next_state = self.dynamics_model(state_tensor, action_tensor)
            dynamics_loss = F.mse_loss(predicted_next_state, next_state_tensor)
            dynamics_loss.backward()
            self.dynamics_optimizer.step()
            
            # Train reward model
            self.reward_optimizer.zero_grad()
            predicted_reward = self.reward_model(state_tensor, action_tensor)
            reward_loss = F.mse_loss(predicted_reward, reward_tensor)
            reward_loss.backward()
            self.reward_optimizer.step()
            
            # Update episode loss metrics
            episode_dynamics_loss += dynamics_loss.item()
            episode_reward_loss += reward_loss.item()
        
        # Average losses over steps in episode
        num_steps = len(episode_data)
        episode_dynamics_loss /= num_steps
        episode_reward_loss /= num_steps
        
        return episode_dynamics_loss, episode_reward_loss
    
    def save_hyperparameters(self):
        """Save hyperparameters to a JSON file."""
        hparams = {
            'model_type': self.model_type,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'state_embed_dim': self.state_embed_dim,
            'action_embed_dim': self.action_embed_dim,
            'num_actions': self.num_actions,
            'dynamics_hidden_dim': self.dynamics_hidden_dim,
            'reward_hidden_dim': self.reward_hidden_dim,
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'device': str(self.device),
            'num_workers': self.num_workers
        }
        
        with open(self.output_dir / 'hyperparameters.json', 'w') as f:
            json.dump(hparams, f, indent=2)
    
    def train_dynamics(self, num_epochs: int = 10, log_interval: int = 10):
        """
        Train the dynamics predictor.
        
        Args:
            num_epochs: Number of epochs to train
            log_interval: Interval for logging training progress
        """
        if self.dynamics_model is None or self.train_loader is None:
            raise ValueError("Models or data loaders not initialized")
            
        print("Training dynamics predictor...")
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(
                model=self.dynamics_model,
                optimizer=self.dynamics_optimizer,
                criterion=self.dynamics_criterion,
                data_loader=self.train_loader,
                is_dynamics=True,
                log_interval=log_interval
            )
            
            val_loss = self._validate(
                model=self.dynamics_model,
                criterion=self.dynamics_criterion,
                data_loader=self.val_loader,
                is_dynamics=True
            )
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(self.dynamics_model, 'dynamics')
                print(f"New best model saved! Validation loss: {val_loss:.6f}")
    
    def train_reward(self, num_epochs: int = 10, log_interval: int = 10):
        """
        Train the reward predictor.
        
        Args:
            num_epochs: Number of epochs to train
            log_interval: Interval for logging training progress
        """
        if self.reward_model is None or self.train_loader is None:
            raise ValueError("Models or data loaders not initialized")
            
        print("Training reward predictor...")
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(
                model=self.reward_model,
                optimizer=self.reward_optimizer,
                criterion=self.reward_criterion,
                data_loader=self.train_loader,
                is_dynamics=False,
                log_interval=log_interval
            )
            
            val_loss = self._validate(
                model=self.reward_model,
                criterion=self.reward_criterion,
                data_loader=self.val_loader,
                is_dynamics=False
            )
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(self.reward_model, 'reward')
                print(f"New best model saved! Validation loss: {val_loss:.6f}")
                
    def _train_epoch(self, model, optimizer, criterion, data_loader, is_dynamics: bool, log_interval: int):
        """
        Train the model for one epoch.
        
        Args:
            model: Model to train
            optimizer: Optimizer for the model
            criterion: Loss criterion
            data_loader: DataLoader for training data
            is_dynamics: Whether training dynamics (True) or reward (False)
            log_interval: Interval for logging
        
        Returns:
            Average loss for the epoch
        """
        model.train()
        total_loss = 0
        total_batches = 0
        epoch_start_time = time.time()
        
        for batch_idx, (state, action, next_state, reward) in enumerate(data_loader):
            # Move data to device
            state = state.to(self.device)
            action = action.to(self.device)
            
            if is_dynamics:
                target = next_state.to(self.device)
                output = model(state, action)
                
                # Sometimes outputs can have different shapes due to padding
                # Make sure target and output have the same shape
                if output.shape != target.shape:
                    output = torch.nn.functional.interpolate(
                        output, 
                        size=target.shape[2:], 
                        mode='nearest'
                    )
            else:
                # For reward prediction
                reward = reward.to(self.device)
                if reward.dim() == 1:
                    # Ensure reward is [batch_size, 1]
                    target = reward.unsqueeze(1)
                else:
                    target = reward
                    
                # Get model prediction
                output = model(state, action)
                
                # Make sure output and target have the same shape
                if output.dim() != target.dim():
                    if output.dim() > target.dim():
                        target = target.unsqueeze(-1)
                    else:
                        output = output.unsqueeze(-1)
            
            loss = criterion(output, target)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            total_batches += 1
            
            # Log progress
            if batch_idx % log_interval == 0:
                elapsed = time.time() - epoch_start_time
                print(f"Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.6f} | Time: {elapsed:.2f}s")
        
        # Handle case where no batches were processed
        if total_batches == 0:
            return 0.0
            
        return total_loss / total_batches
    
    def _validate(self, model, criterion, data_loader, is_dynamics: bool):
        """
        Validate the model.
        
        Args:
            model: Model to validate
            criterion: Loss criterion
            data_loader: DataLoader for validation data
            is_dynamics: Whether validating dynamics (True) or reward (False)
        
        Returns:
            Average validation loss
        """
        model.eval()
        total_loss = 0
        total_batches = 0
        
        with torch.no_grad():
            for state, action, next_state, reward in data_loader:
                # Move data to device
                state = state.to(self.device)
                action = action.to(self.device)
                
                if is_dynamics:
                    target = next_state.to(self.device)
                    output = model(state, action)
                    
                    # Sometimes outputs can have different shapes due to padding
                    # Make sure target and output have the same shape
                    if output.shape != target.shape:
                        output = torch.nn.functional.interpolate(
                            output, 
                            size=target.shape[2:], 
                            mode='nearest'
                        )
                else:
                    # For reward prediction
                    reward = reward.to(self.device)
                    if reward.dim() == 1:
                        # Ensure reward is [batch_size, 1]
                        target = reward.unsqueeze(1)
                    else:
                        target = reward
                        
                    # Get model prediction
                    output = model(state, action)
                    
                    # Make sure output and target have the same shape
                    if output.dim() != target.dim():
                        if output.dim() > target.dim():
                            target = target.unsqueeze(-1)
                        else:
                            output = output.unsqueeze(-1)
                
                loss = criterion(output, target)
                total_loss += loss.item()
                total_batches += 1
        
        # Handle case where no batches were processed
        if total_batches == 0:
            return 0.0
            
        return total_loss / total_batches
    
    def save_model(self, model, model_name: str):
        """
        Save a model.
        
        Args:
            model: Model to save
            model_name: Name of the model (e.g. 'dynamics', 'reward')
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save model
        model_path = self.output_dir / f"{model_name}_model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str):
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model to load ('dynamics' or 'reward')
        """
        model_path = self.output_dir / f"{model_name}_model.pt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        if model_name == 'dynamics':
            if self.dynamics_model is None:
                raise ValueError("Dynamics model not initialized. Initialize models first.")
            self.dynamics_model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded dynamics model from {model_path}")
        elif model_name == 'reward':
            if self.reward_model is None:
                raise ValueError("Reward model not initialized. Initialize models first.")
            self.reward_model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded reward model from {model_path}")
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def train_all(self, dynamics_epochs: int = 10, reward_epochs: int = 10, log_interval: int = 10):
        """
        Train both dynamics and reward models.
        
        Args:
            dynamics_epochs: Number of epochs to train dynamics model
            reward_epochs: Number of epochs to train reward model
            log_interval: Interval for logging
        """
        # Train dynamics model
        self.train_dynamics(num_epochs=dynamics_epochs, log_interval=log_interval)
        
        # Train reward model
        self.train_reward(num_epochs=reward_epochs, log_interval=log_interval) 