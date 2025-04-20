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
import logging # Import logging
from tqdm import tqdm # Import tqdm

# Attempt to import wandb, but don't fail if it's not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from .state_encoder import GridStateEncoder, VectorStateEncoder, StateEncoderNetwork, VectorEncoderNetwork
from .dynamics_predictor import GridDynamicsPredictor, VectorDynamicsPredictor, SharedEncoderDynamicsPredictor
from .reward_predictor import GridRewardPredictor, VectorRewardPredictor, SharedEncoderRewardPredictor
from .data_processor import get_overcooked_dataloaders

# Get a logger for this module
logger = logging.getLogger(__name__)

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
        val_ratio=0.1,
        test_ratio=0.1,
        seed=None,
        env=None,
        state_encoder=None,
        gamma=0.99,
        wandb_run=None, # Pass wandb run object (optional)
        disable_artifacts=True # Disable WandB artifacts by default
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
        self.val_ratio = float(val_ratio) if not isinstance(val_ratio, float) else val_ratio
        self.test_ratio = float(test_ratio) if not isinstance(test_ratio, float) else test_ratio
        self.seed = seed
        self.disable_artifacts = disable_artifacts
        
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
        
        # WandB setup
        self.wandb_run = wandb_run
        self.use_wandb = WANDB_AVAILABLE and (self.wandb_run is not None)
        if self.use_wandb:
             logger.info("WandB logging enabled within trainer.")
        else:
             logger.info("WandB logging disabled within trainer (wandb not installed or run not provided).")

        # Print parameters for debugging using logger
        logger.info(f"Initializing PLDMTrainer with: lr={self.lr}, batch_size={self.batch_size}, model_type={self.model_type}")
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        logger.info(f"Trainer using device: {self.device}")
        
        # Initialize state encoder based on model type
        if state_encoder is not None:
            self.state_encoder = state_encoder
            logger.info("Using provided state encoder.")
        elif model_type == "grid":
            self.state_encoder = GridStateEncoder(grid_height=self.grid_height, grid_width=self.grid_width)
            logger.info(f"Initialized GridStateEncoder (grid_height={self.grid_height}, grid_width={self.grid_width})")
        else:  # model_type == "vector"
            self.state_encoder = VectorStateEncoder(grid_height=self.grid_height, grid_width=self.grid_width)
            logger.info(f"Initialized VectorStateEncoder (grid_height={self.grid_height}, grid_width={self.grid_width} used for normalization)")
        
        # Initialize models and optimizers
        self.dynamics_model = None
        self.reward_model = None
        self.dynamics_optimizer = None
        self.reward_optimizer = None
        self.dynamics_criterion = nn.MSELoss()
        self.reward_criterion = nn.MSELoss()
        
        # Load and prepare data if path is provided
        if data_path:
            logger.info("Loading data...")
            self.train_loader, self.val_loader, self.test_loader = get_overcooked_dataloaders(
                data_path=data_path,
                state_encoder_type=model_type,
                batch_size=batch_size,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                max_samples=max_samples,
                num_workers=num_workers,
                seed=self.seed
            )
            logger.info("Data loaded.")
            
            # Initialize models based on a sample from the dataset
            logger.info("Initializing models...")
            self._initialize_models()
        else:
            logger.warning("No data_path provided. Models and data loaders will not be initialized.")
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
        
        # Tracking metrics
        self.dynamics_losses = []
        self.reward_losses = []
        
    def _initialize_models(self):
        """Initialize the dynamics and reward models based on model type."""
        if not self.train_loader:
             logger.error("Cannot initialize models without a data loader.")
             raise ValueError("Data loader not available for model initialization.")
             
        # Get a sample batch to determine dimensions
        try:
            sample_batch = next(iter(self.train_loader))
            state, action, _, _ = sample_batch
            logger.info("Got sample batch for model initialization.")
        except StopIteration:
             logger.error("Data loader is empty. Cannot initialize models.")
             raise ValueError("Data loader is empty.")
        except Exception as e:
            logger.error(f"Error getting sample batch: {e}")
            raise
        
        # Get state shape
        if self.model_type == "grid":
            batch_size, num_channels, grid_height, grid_width = state.shape
            
            # Update grid dimensions if not explicitly set
            if self.grid_height is None:
                self.grid_height = grid_height
            if self.grid_width is None:
                self.grid_width = grid_width
                
            logger.info(f"Initializing grid models with dimensions: channels={num_channels}, height={self.grid_height}, width={self.grid_width}")
            
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
            logger.info(f"Initializing vector models with state_dim={state_dim}")
            
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
        logger.info("Optimizers initialized.")
        
        # Print the size of the model to verify
        total_dynamics_params = sum(p.numel() for p in self.dynamics_model.parameters() if p.requires_grad)
        total_reward_params = sum(p.numel() for p in self.reward_model.parameters() if p.requires_grad)
        
        logger.info(f"Initialized {self.model_type} models.")
        logger.debug(f"Dynamics model: {self.dynamics_model}")
        logger.debug(f"Reward model: {self.reward_model}")
        logger.info(f"Total trainable parameters - Dynamics: {total_dynamics_params:,}, Reward: {total_reward_params:,}")

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
            logger.error("Cannot train dynamics: Models or data loaders not initialized")
            raise ValueError("Models or data loaders not initialized")
            
        logger.info(f"Training dynamics predictor for {num_epochs} epochs...")
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_loss = self._train_epoch(
                model=self.dynamics_model,
                optimizer=self.dynamics_optimizer,
                criterion=self.dynamics_criterion,
                data_loader=self.train_loader,
                is_dynamics=True,
                log_interval=log_interval,
                epoch=epoch # Pass epoch for wandb logging
            )
            
            val_loss = self._validate(
                model=self.dynamics_model,
                criterion=self.dynamics_criterion,
                data_loader=self.val_loader,
                is_dynamics=True,
                epoch=epoch # Pass epoch for wandb logging
            )
            
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Duration: {epoch_duration:.2f}s")
            
            # Log metrics to WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "dynamics_train_loss": train_loss,
                    "dynamics_val_loss": val_loss,
                    "dynamics_epoch_duration_sec": epoch_duration
                })
            
            # Save best model
            if val_loss < best_val_loss:
                logger.info(f"New best dynamics model found! Val loss improved from {best_val_loss:.6f} to {val_loss:.6f}")
                best_val_loss = val_loss
                self.save_model(self.dynamics_model, 'dynamics')
                
        total_training_time = time.time() - start_time
        logger.info(f"Finished training dynamics model. Total time: {total_training_time:.2f}s")
    
    def train_reward(self, num_epochs: int = 10, log_interval: int = 10):
        """
        Train the reward predictor.
        
        Args:
            num_epochs: Number of epochs to train
            log_interval: Interval for logging training progress
        """
        if self.reward_model is None or self.train_loader is None:
            logger.error("Cannot train reward: Models or data loaders not initialized")
            raise ValueError("Models or data loaders not initialized")
            
        logger.info(f"Training reward predictor for {num_epochs} epochs...")
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_loss = self._train_epoch(
                model=self.reward_model,
                optimizer=self.reward_optimizer,
                criterion=self.reward_criterion,
                data_loader=self.train_loader,
                is_dynamics=False,
                log_interval=log_interval,
                epoch=epoch # Pass epoch for wandb logging
            )
            
            val_loss = self._validate(
                model=self.reward_model,
                criterion=self.reward_criterion,
                data_loader=self.val_loader,
                is_dynamics=False,
                epoch=epoch # Pass epoch for wandb logging
            )
            
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Duration: {epoch_duration:.2f}s")
            
            # Log metrics to WandB
            if self.use_wandb:
                 wandb.log({
                    "epoch": epoch + 1,
                    "reward_train_loss": train_loss,
                    "reward_val_loss": val_loss,
                    "reward_epoch_duration_sec": epoch_duration
                })

            # Save best model
            if val_loss < best_val_loss:
                logger.info(f"New best reward model found! Val loss improved from {best_val_loss:.6f} to {val_loss:.6f}")
                best_val_loss = val_loss
                self.save_model(self.reward_model, 'reward')
                
        total_training_time = time.time() - start_time
        logger.info(f"Finished training reward model. Total time: {total_training_time:.2f}s")
                
    def _train_epoch(self, model, optimizer, criterion, data_loader, is_dynamics: bool, log_interval: int, epoch: int):
        """
        Train the model for one epoch.
        
        Args:
            model: Model to train
            optimizer: Optimizer for the model
            criterion: Loss criterion
            data_loader: DataLoader for training data
            is_dynamics: Whether training dynamics (True) or reward (False)
            log_interval: Interval for logging
            epoch: Current epoch number (for logging)
        
        Returns:
            Average loss for the epoch
        """
        model.train()
        total_loss = 0
        total_batches = 0
        epoch_start_time = time.time()
        
        # Use tqdm for progress bar
        data_iterator = tqdm(data_loader, desc=f"Epoch {epoch+1} Train", leave=False, unit="batch")

        for batch_idx, (state, action, next_state, reward) in enumerate(data_iterator):
            # Move data to device
            state = state.to(self.device)
            action = action.to(self.device)
            
            if is_dynamics:
                target = next_state.to(self.device)
                output = model(state, action)
                
                # Sometimes outputs can have different shapes due to padding
                if output.shape != target.shape:
                    # Resize output to match target (often needed for dynamics)
                    output = torch.nn.functional.interpolate(
                        output, size=target.shape[2:], mode='nearest')
            else:
                # For reward prediction
                reward = reward.to(self.device)
                target = reward.unsqueeze(1) if reward.dim() == 1 else reward
                output = model(state, action)
                if output.dim() == 1: output = output.unsqueeze(1) # Ensure output is [batch, 1]
                # Ensure shapes match (less common for reward, but possible)
                if output.shape != target.shape:
                    # This case is less expected for reward, log a warning
                    logger.warning(f"Reward output shape {output.shape} mismatch with target {target.shape} in batch {batch_idx}. Attempting target reshape.")
                    # Try reshaping target first
                    try:
                        target = target.view_as(output)
                    except RuntimeError:
                         logger.error("Could not reshape reward target to match output. Skipping loss calculation for this batch.")
                         continue # Skip batch if shapes irreconcilable
            
            # Calculate loss
            try:
                loss = criterion(output, target)
            except Exception as loss_err:
                logger.error(f"Error computing loss for batch {batch_idx}: {loss_err}. Output: {output.shape}, Target: {target.shape}")
                continue # Skip batch if loss calculation fails
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            batch_loss = loss.item()
            total_loss += batch_loss
            total_batches += 1
            
            # Update tqdm progress bar
            data_iterator.set_postfix(loss=f"{batch_loss:.6f}")
            
            # Log progress (less frequently with tqdm)
            if batch_idx % log_interval == 0 and log_interval > 0:
                # Log less verbosely when using tqdm
                pass 
                # logger.debug(f"Epoch {epoch+1} Batch {batch_idx}/{len(data_loader)} | Loss: {batch_loss:.6f}")
        
        data_iterator.close()
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        logger.debug(f"Epoch {epoch+1} Average Train Loss: {avg_loss:.6f}")
        return avg_loss
    
    def _validate(self, model, criterion, data_loader, is_dynamics: bool, epoch: int):
        """
        Validate the model.
        
        Args:
            model: Model to validate
            criterion: Loss criterion
            data_loader: DataLoader for validation data
            is_dynamics: Whether validating dynamics (True) or reward (False)
            epoch: Current epoch number (for logging)
        
        Returns:
            Average validation loss
        """
        model.eval()
        total_loss = 0
        total_batches = 0
        
        # Use tqdm for progress bar
        data_iterator = tqdm(data_loader, desc=f"Epoch {epoch+1} Validate", leave=False, unit="batch")
        
        with torch.no_grad():
            for batch_idx, (state, action, next_state, reward) in enumerate(data_iterator):
                state = state.to(self.device)
                action = action.to(self.device)
                
                if is_dynamics:
                    target = next_state.to(self.device)
                    output = model(state, action)
                    if output.shape != target.shape:
                         output = torch.nn.functional.interpolate(output, size=target.shape[2:], mode='nearest')
                else:
                    reward = reward.to(self.device)
                    target = reward.unsqueeze(1) if reward.dim() == 1 else reward
                    output = model(state, action)
                    if output.dim() == 1: output = output.unsqueeze(1)
                    if output.shape != target.shape:
                         logger.warning(f"Reward output shape {output.shape} mismatch target {target.shape} in validation batch {batch_idx}.")
                         try:
                             target = target.view_as(output)
                         except RuntimeError:
                             logger.error("Validation: Could not reshape reward target. Skipping batch.")
                             continue
                
                try:
                    loss = criterion(output, target)
                    total_loss += loss.item()
                    total_batches += 1
                    data_iterator.set_postfix(loss=f"{loss.item():.6f}")
                except Exception as loss_err:
                    logger.error(f"Error computing validation loss for batch {batch_idx}: {loss_err}")
                    continue

        data_iterator.close()
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        logger.debug(f"Epoch {epoch+1} Average Validation Loss: {avg_loss:.6f}")
        return avg_loss
    
    def save_model(self, model, model_name: str):
        """
        Save a model state dict and potentially log to WandB as an artifact.
        
        Args:
            model: Model to save
            model_name: Name of the model (e.g. 'dynamics', 'reward')
        """
        os.makedirs(self.output_dir, exist_ok=True)
        model_path = self.output_dir / f"{model_name}_model.pt"
        
        try:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

            # Log model artifact to WandB if enabled and artifacts are not disabled
            if self.use_wandb and not self.disable_artifacts:
                 try:
                     artifact_name = f"{model_name}_model"
                     artifact = wandb.Artifact(artifact_name, type='model')
                     artifact.add_file(str(model_path))
                     self.wandb_run.log_artifact(artifact)
                     logger.info(f"Logged {model_name} model artifact to WandB.")
                 except Exception as wb_err:
                     logger.error(f"Failed to log model artifact to WandB: {wb_err}")
            elif self.use_wandb and self.disable_artifacts:
                logger.info(f"WandB artifact logging disabled. Model saved locally only.")
        except Exception as e:
            logger.error(f"Failed to save model {model_name} to {model_path}: {e}")
    
    def load_model(self, model_name: str):
        """
        Load a saved model state dict.
        
        Args:
            model_name: Name of the model to load ('dynamics' or 'reward')
        """
        model_path = self.output_dir / f"{model_name}_model.pt"
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        target_model = None
        if model_name == 'dynamics':
            if self.dynamics_model is None:
                 logger.error("Dynamics model not initialized before loading.")
                 raise ValueError("Dynamics model not initialized.")
            target_model = self.dynamics_model
        elif model_name == 'reward':
            if self.reward_model is None:
                 logger.error("Reward model not initialized before loading.")
                 raise ValueError("Reward model not initialized.")
            target_model = self.reward_model
        else:
            logger.error(f"Unknown model name for loading: {model_name}")
            raise ValueError(f"Unknown model name: {model_name}")
            
        try:
            # Use the custom loader if model has dynamic layers (Grid models)
            if isinstance(target_model, (GridDynamicsPredictor, GridRewardPredictor)):
                 from solution.test_pldm import load_model_with_dynamic_layers # Avoid circular import if possible
                 load_model_with_dynamic_layers(target_model, model_path, self.device)
            else:
                 # Standard loading for vector models or others
                 target_model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded {model_name} model from {model_path}")
        except Exception as e:
             logger.error(f"Error loading model state dict for {model_name} from {model_path}: {e}")
             raise # Re-raise the exception after logging
    
    def evaluate_test_set(self, model_type='both'):
        """
        Evaluate models on the test set.
        
        Args:
            model_type: Which model to evaluate ('dynamics', 'reward', or 'both')
        
        Returns:
            Dictionary of test metrics
        """
        if self.test_loader is None:
            logger.error("Cannot evaluate: Test loader not initialized")
            return {}

        test_metrics = {}
        
        if model_type in ['dynamics', 'both']:
            if self.dynamics_model is None:
                logger.error("Cannot evaluate dynamics: Model not initialized")
            else:
                dynamics_loss = self._validate(
                    model=self.dynamics_model,
                    criterion=self.dynamics_criterion,
                    data_loader=self.test_loader,
                    is_dynamics=True,
                    epoch=-1  # -1 indicates test set evaluation
                )
                test_metrics['dynamics_test_loss'] = dynamics_loss
                logger.info(f"Dynamics Test Loss: {dynamics_loss:.6f}")
                
                # Log to WandB if enabled
                if self.use_wandb:
                    wandb.log({"dynamics_test_loss": dynamics_loss})
        
        if model_type in ['reward', 'both']:
            if self.reward_model is None:
                logger.error("Cannot evaluate reward: Model not initialized")
            else:
                reward_loss = self._validate(
                    model=self.reward_model,
                    criterion=self.reward_criterion,
                    data_loader=self.test_loader,
                    is_dynamics=False,
                    epoch=-1  # -1 indicates test set evaluation
                )
                test_metrics['reward_test_loss'] = reward_loss
                logger.info(f"Reward Test Loss: {reward_loss:.6f}")
                
                # Log to WandB if enabled
                if self.use_wandb:
                    wandb.log({"reward_test_loss": reward_loss})
        
        return test_metrics
    
    def train_all(self, dynamics_epochs: int = 10, reward_epochs: int = 10, log_interval: int = 10):
        """
        Train both dynamics and reward models.
        
        Args:
            dynamics_epochs: Number of epochs to train dynamics model
            reward_epochs: Number of epochs to train reward model
            log_interval: Interval for logging
        """
        logger.info("Starting training for both dynamics and reward models.")
        
        # Train dynamics model
        self.train_dynamics(num_epochs=dynamics_epochs, log_interval=log_interval)
        
        # Train reward model
        self.train_reward(num_epochs=reward_epochs, log_interval=log_interval)
        
        # Evaluate on test set after training
        logger.info("Evaluating final models on test set...")
        test_metrics = self.evaluate_test_set(model_type='both')
        
        logger.info("Finished training both models.") 