import os
import sys
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
import inspect

# Attempt to import wandb, but don't fail if it's not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from .state_encoder import GridStateEncoder, VectorStateEncoder, StateEncoderNetwork
from .dynamics_predictor import GridDynamicsPredictor, VectorDynamicsPredictor, SharedEncoderDynamicsPredictor
from .reward_predictor import GridRewardPredictor, VectorRewardPredictor, SharedEncoderRewardPredictor
from .data_processor import get_overcooked_dataloaders
from .objectives import get_loss_function, MSELoss, VICRegLoss, VICRegConfig, MSELossInfo, VICRegLossInfo # Corrected info object imports
from .predictors import TransformerPredictor, create_dynamics_predictor, create_reward_predictor # Keep create functions

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
        disable_artifacts=True, # Disable WandB artifacts by default
        config=None,  # Added configuration parameter
        teacher_forcing_ratio=0.5,  # Added teacher forcing ratio
        weight_decay=0.0,  # Added weight decay parameter
        max_norm=0.0,  # Added gradient clipping parameter
        train_loader=None,
        val_loader=None,
        test_loader=None
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
        self.config = config or {}  # Store config or use empty dict
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.weight_decay = weight_decay
        self.max_norm = max_norm
        
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
            
            # Initialize CNN encoder if specified in config
            if config and config.get("model", {}).get("encoder_type") == "cnn":
                logger.info("Using CNNStateEncoderNetwork for state encoding")
                self.cnn_encoder = StateEncoderNetwork(
                    input_channels=32,  # Default number of channels in GridStateEncoder
                    state_embed_dim=config.get("model", {}).get("state_embed_dim", 128),
                    grid_height=self.grid_height,
                    grid_width=self.grid_width
                ).to(self.device)
            else:
                self.cnn_encoder = None
        else:  # model_type == "vector"
            self.state_encoder = VectorStateEncoder(grid_height=self.grid_height, grid_width=self.grid_width)
            logger.info(f"Initialized VectorStateEncoder (grid_height={self.grid_height}, grid_width={self.grid_width} used for normalization)")
        
        # Initialize models and optimizers
        self.dynamics_model = None
        self.reward_model = None
        self.dynamics_optimizer = None
        self.reward_optimizer = None
        
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
            state, action, next_state, reward = sample_batch
            logger.info("Got sample batch for model initialization.")

            # Information about grid size (if applicable)
            if self.model_type == "grid":
                if len(state.shape) < 4:
                    logger.error(f"Expected grid state (4D tensor) but got shape {state.shape}. Check data loading or model_type config.")
                    raise ValueError(f"Incorrect state shape for grid model: {state.shape}")
                batch_size, num_channels, grid_height, grid_width = state.shape
                logger.info(f"Grid dimensions: {grid_height}x{grid_width} with {num_channels} channels")

                # Set grid dimensions if not already specified in config
                if self.grid_height is None or self.grid_width is None:
                    self.grid_height = grid_height
                    self.grid_width = grid_width
                    logger.info(f"Updated grid dimensions to {self.grid_height}x{self.grid_width}")
                elif self.grid_height != grid_height or self.grid_width != grid_width:
                    logger.warning(f"Config grid dims ({self.grid_height}x{self.grid_width}) mismatch data ({grid_height}x{grid_width}). Using data dims.")
                    self.grid_height = grid_height
                    self.grid_width = grid_width

                input_dim = num_channels # For GridStateEncoder or initial CNN layer
                state_repr_dim = grid_height * grid_width * num_channels # Initial state representation dimension (flattened grid)

            elif self.model_type == "vector":
                if len(state.shape) != 2:
                     logger.error(f"Expected vector state (2D tensor) but got shape {state.shape}.")
                     raise ValueError(f"Incorrect state shape for vector model: {state.shape}")
                batch_size, state_dim = state.shape
                input_dim = state_dim
                state_repr_dim = state_dim # Vector representation is just the input dim
                logger.info(f"Vector state dimension: {state_dim}")
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

            # Get action dimension
            if len(action.shape) == 1: # Assume scalar actions need embedding
                 action_dim = 1 # Will be embedded later
                 self.num_actions = action.max().item() + 1 if action.numel() > 0 else self.num_actions # Infer if possible
                 logger.info(f"Scalar action detected. Inferred num_actions: {self.num_actions}")
            elif len(action.shape) == 2:
                 action_dim = action.shape[1]
                 logger.info(f"Vector action dimension: {action_dim}")
            else:
                 raise ValueError(f"Unsupported action shape: {action.shape}")

        except StopIteration:
            logger.error("Training loader is empty, cannot initialize models.")
            raise ValueError("Cannot initialize models from empty data loader.")
        except Exception as e:
            logger.error(f"Error getting sample batch for model init: {e}", exc_info=True)
            raise

        # --- State Encoder Initialization ---
        model_config = self.config.get("model", {})
        if self.state_encoder is None: # Only initialize if not provided externally
            if self.model_type == "grid":
                # Always start with GridStateEncoder for grid models
                self.state_encoder = GridStateEncoder(grid_height=self.grid_height, grid_width=self.grid_width)
                logger.info(f"Initialized GridStateEncoder (grid_height={self.grid_height}, grid_width={self.grid_width})")

                # Optionally add CNN on top
                if self.config.get("model", {}).get("encoder_type") == "cnn":
                    logger.info("Using CNNStateEncoderNetwork for state encoding")
                    cnn_config = {
                        "input_channels": model_config.get("in_channels", 32), # Ensure this matches grid encoder output
                        "hidden_channels": model_config.get("hidden_channels", [16, 32, 64]),
                        "fc_dims": model_config.get("fc_dims", [256, 128]),
                        "output_dim": self.state_embed_dim, # Use state_embed_dim from config
                        "grid_height": self.grid_height,
                        "grid_width": self.grid_width,
                        "activation": model_config.get("activation", "relu"),
                        "use_layer_norm": model_config.get("use_layer_norm", False),
                        "dynamic_net": False # Initialize immediately
                    }
                    self.cnn_encoder = StateEncoderNetwork(**cnn_config)
                    # The final state representation dim comes from the CNN
                    state_repr_dim = self.state_embed_dim
                    logger.info(f"Initialized CNNStateEncoderNetwork with output dimension: {state_repr_dim}")
                else:
                    # If not CNN, the representation is the flattened output of GridStateEncoder
                    # Calculate this based on GridStateEncoder logic (if needed)
                    # For simplicity, assume predictors handle raw grid input directly if no CNN
                    state_repr_dim = self.grid_height * self.grid_width * model_config.get("in_channels", 32) # Example placeholder
                    logger.info(f"Using raw grid state representation (dim: {state_repr_dim})")

            elif self.model_type == "vector":
                self.state_encoder = VectorStateEncoder(input_dim=input_dim) # Simple identity or MLP encoder
                state_repr_dim = self.state_encoder.output_dim # Get actual output dim
                logger.info(f"Initialized VectorStateEncoder with output dimension: {state_repr_dim}")
        else:
             logger.info("Using provided state encoder.")
             # Need to determine the output dimension of the provided encoder
             # This is tricky without knowing the encoder type, assume it has an 'output_dim' attribute
             if hasattr(self.state_encoder, 'output_dim'):
                 state_repr_dim = self.state_encoder.output_dim
             else:
                 # Attempt to infer by passing dummy data (might be risky)
                 try:
                     dummy_output = self.state_encoder(state.to(self.device))
                     state_repr_dim = dummy_output.shape[-1]
                     logger.info(f"Inferred state encoder output dimension: {state_repr_dim}")
                 except Exception as e:
                     logger.error(f"Could not determine output dimension of provided state encoder: {e}")
                     # Fallback or raise error
                     state_repr_dim = self.state_embed_dim # Fallback to config value
                     logger.warning(f"Falling back to config state_embed_dim: {state_repr_dim}")


        logger.info(f"Final state representation dimension for predictors: {state_repr_dim}")
        logger.info(f"Action dimension: {action_dim}")


        # --- Predictor Initialization ---
        dyn_pred_config = self.get_predictor_config('dynamics')
        rew_pred_config = self.get_predictor_config('reward')

        # Create Dynamics Predictor
        if self.config.get("training", {}).get("train_dynamics", True):
            try:
                self.dynamics_model = create_dynamics_predictor(
                    config=dyn_pred_config,
                    input_dim=state_repr_dim, # Use final state repr dim
                    output_dim=state_repr_dim # Dynamics typically predicts next state repr
                ).to(self.device)
                logger.info(f"Initialized dynamics model: {type(self.dynamics_model).__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize dynamics predictor: {e}", exc_info=True)
                raise

        # Create Reward Predictor
        if self.config.get("training", {}).get("train_reward", True):
            try:
                self.reward_model = create_reward_predictor(
                    config=rew_pred_config,
                    input_dim=state_repr_dim # Use final state repr dim
                ).to(self.device)
                logger.info(f"Initialized reward model: {type(self.reward_model).__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize reward predictor: {e}", exc_info=True)
                raise

        # Final initialization step: Optimizers and Loss functions
        self._initialize_optimizers()
        self._initialize_loss_functions()

        logger.info("Models, optimizers, and loss functions initialized.")

    def get_predictor_config(self, model_type='dynamics'):
        """
        Get the predictor configuration from the config dictionary.
        
        Args:
            model_type: 'dynamics' or 'reward'
            
        Returns:
            Configuration dictionary for the predictor
        """
        model_config = self.config.get("model", {})
        
        if model_type == 'dynamics':
            predictor_config = model_config.get("dynamics_predictor", {})
            # Set defaults based on trainer attributes if keys missing in config
            predictor_config.setdefault("predictor_type", self.model_type) # Default to grid/vector
            predictor_config.setdefault("hidden_size", self.dynamics_hidden_dim)
            predictor_config.setdefault("num_actions", self.num_actions)
            predictor_config.setdefault("action_embed_dim", self.action_embed_dim)
            # Add other dynamics-specific defaults from model_config if needed
        elif model_type == 'reward':
            predictor_config = model_config.get("reward_predictor", {})
            # Set defaults
            predictor_config.setdefault("predictor_type", self.model_type) # Default to grid/vector
            predictor_config.setdefault("hidden_size", self.reward_hidden_dim)
            predictor_config.setdefault("num_actions", self.num_actions)
            predictor_config.setdefault("action_embed_dim", self.action_embed_dim)
            # Add other reward-specific defaults from model_config if needed
        else:
            raise ValueError(f"Unknown model_type for predictor config: {model_type}")

        # Add general params that might be needed by predictors
        predictor_config.setdefault("activation", model_config.get("activation", "relu"))
        predictor_config.setdefault("use_layer_norm", model_config.get("use_layer_norm", False))
        predictor_config.setdefault("teacher_forcing_ratio", model_config.get("teacher_forcing_ratio", 0.0)) # For RNN/Transformers

        return predictor_config


    def _initialize_optimizers(self):
        """Initialize optimizers for the models."""
        if self.dynamics_model:
            self.dynamics_optimizer = self._create_optimizer(self.dynamics_model)
            logger.info(f"Initialized dynamics optimizer: {self.config.get('optimizer', 'adam')}")
        if self.reward_model:
            self.reward_optimizer = self._create_optimizer(self.reward_model)
            logger.info(f"Initialized reward optimizer: {self.config.get('optimizer', 'adam')}")
        # Add CNN encoder parameters if it exists and is trainable
        if self.cnn_encoder and self.config.get("training", {}).get("train_dynamics", True): # Assuming CNN trained with dynamics
             # Create a separate optimizer or add params to dynamics optimizer?
             # Adding to dynamics optimizer for simplicity
             logger.info("Adding CNN encoder parameters to dynamics optimizer.")
             self.dynamics_optimizer.add_param_group({'params': self.cnn_encoder.parameters()})


    def _create_optimizer(self, model):
        """Helper function to create an optimizer instance."""
        optimizer_type = self.config.get("optimizer", "adam").lower()
        if optimizer_type == "adam":
            return optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_type == "adamw":
             return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_type == "sgd":
            return optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            logger.warning(f"Unsupported optimizer type: {optimizer_type}. Defaulting to Adam.")
            return optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def _initialize_loss_functions(self):
        """Initialize loss functions based on configuration."""
        loss_config = self.config.get("loss", {})
        dynamics_loss_type = loss_config.get("dynamics_loss", "mse")
        reward_loss_type = loss_config.get("reward_loss", "mse")
        
        logger.info(f"Initializing loss functions - Dynamics: {dynamics_loss_type}, Reward: {reward_loss_type}")
        
        # Initialize dynamics loss
        if self.config.get("training", {}).get("train_dynamics", True):
            if dynamics_loss_type.lower() == "vicreg":
                vicreg_config_dict = loss_config.get("vicreg", {})
                # Create VICRegConfig object
                vicreg_config_obj = VICRegConfig(**vicreg_config_dict)

                # Determine projector input dimension
                if self.model_type == "grid":
                    proj_input_dim = self.state_embed_dim if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None else self.grid_height * self.grid_width * 32 # Placeholder channel num
                else: # vector
                    proj_input_dim = self.state_embed_dim # Assume state_embed_dim is set

                self.dynamics_criterion = VICRegLoss(
                    config=vicreg_config_obj,
                    input_dim=proj_input_dim
                )
                logger.info(f"Using VICRegLoss for dynamics with config: {vicreg_config_obj}")
            elif dynamics_loss_type.lower() == "mse":
                self.dynamics_criterion = MSELoss()
                logger.info("Using MSELoss for dynamics.")
            else:
                raise ValueError(f"Unsupported dynamics_loss type: {dynamics_loss_type}")

        # Initialize reward loss (assuming reward is always scalar or vector, MSE is common)
        # Determine input dimension for VICReg projector if needed for reward
        reward_projector_input_dim = None
        if reward_loss_type.lower() == "vicreg" and self.reward_model:
             # Reward model outputs scalar, VICReg typically needs embeddings.
             # This setup seems unlikely. VICReg usually compares embeddings.
             # If reward model outputs embeddings before final layer, use that dim.
             logger.warning("Using VICRegLoss for scalar reward prediction is unusual. Ensure the reward model outputs suitable embeddings.")
             # Try to infer based on reward model structure or use a default
             reward_projector_input_dim = self.state_embed_dim # Default/Placeholder

        if self.config.get("training", {}).get("train_reward", True):
            if reward_loss_type.lower() == "vicreg":
                vicreg_config_dict = loss_config.get("vicreg", {})
                vicreg_config_obj = VICRegConfig(**vicreg_config_dict)

                if reward_projector_input_dim is None:
                     logger.error("Cannot initialize VICRegLoss for reward without projector_input_dim.")
                     raise ValueError("VICReg projector input dimension could not be determined for reward.")

                self.reward_criterion = VICRegLoss(
                    config=vicreg_config_obj,
                    input_dim=reward_projector_input_dim # This might need adjustment based on reward model
                )
                logger.info(f"Using VICRegLoss for reward with config: {vicreg_config_obj}")
            elif reward_loss_type.lower() == "mse":
                self.reward_criterion = MSELoss()
                logger.info("Using MSELoss for reward.")
            else:
                raise ValueError(f"Unsupported reward_loss type: {reward_loss_type}")


    def train_batch(self, model_type, batch, epoch, batch_idx, log_interval=10):
        """
        Train on a single batch of data.
        
        Args:
            model_type: 'dynamics' or 'reward'
            batch: Tuple of (state, action, next_state, reward)
            epoch: Current epoch number (for logging)
            batch_idx: Current batch index (for logging)
            log_interval: How often to log training details
        
        Returns:
            Loss value for this batch (float)
        """
        # Select the appropriate model, optimizer, and criterion
        if model_type == 'dynamics':
            model = self.dynamics_model
            optimizer = self.dynamics_optimizer
            criterion = self.dynamics_criterion
        elif model_type == 'reward':
            model = self.reward_model
            optimizer = self.reward_optimizer
            criterion = self.reward_criterion
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        if model is None or optimizer is None or criterion is None:
            logger.warning(f"Skipping training batch for {model_type}: Model, optimizer, or criterion not initialized.")
            return 0.0 # Or None?

        model.train()
        # Ensure CNN encoder is also in train mode if used
        if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None and model_type == 'dynamics':
             self.cnn_encoder.train()

        state, action, next_state, reward = batch
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)

        # --- Forward Pass ---
        try:
            # Handle state encoding (optional CNN)
            if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None and self.model_type == 'grid':
                state_repr = self.cnn_encoder(state)
                next_state_repr = self.cnn_encoder(next_state) # Encode next state too if needed by loss
            else:
                state_repr = state # Use raw state if no CNN
                next_state_repr = next_state

            # Get model output (predicted next state repr or reward)
            # Handle different model call signatures if necessary
            if isinstance(model, TransformerPredictor):
                # Transformer might use teacher forcing
                #  teacher_forcing_ratio = self.config.get("model", {}).get("teacher_forcing_ratio", 0.0)
                #  output = model(state_repr, action, target_seq=next_state_repr, teacher_forcing_ratio=teacher_forcing_ratio)
                 # Transformer no longer uses target_seq or teacher_forcing_ratio directly in forward
                 output = model(state_repr, action)
            else:
                 output = model(state_repr, action)

            # Determine the target based on model_type and loss
            if model_type == 'dynamics':
                # Dynamics model predicts next state representation
                target = next_state_repr
            else: # reward
                # Reward model predicts scalar reward
                target = reward.float().unsqueeze(1) # Ensure target is [batch_size, 1] float

            # --- Loss Computation ---
            # Both MSELoss and VICRegLoss return a single *Info object
            loss_info = criterion(output, target)
            loss = loss_info.total_loss # Extract the actual scalar loss from the info object

        except Exception as e:
            logger.error(f"Error during forward pass or loss computation for {model_type}: {e}", exc_info=True)
            logger.error(f"Input state shape: {state.shape}, action shape: {action.shape}")
            if 'state_repr' in locals(): logger.error(f"State representation shape: {state_repr.shape}")
            if 'output' in locals(): logger.error(f"Model output shape: {output.shape}")
            if 'target' in locals(): logger.error(f"Target shape: {target.shape}")
            raise # Re-raise the exception to halt training

        # --- Backpropagation ---
        try:
            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
                if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None and model_type == 'dynamics':
                     torch.nn.utils.clip_grad_norm_(self.cnn_encoder.parameters(), self.max_norm)

            optimizer.step()
        except Exception as e:
            logger.error(f"Error during backward pass or optimizer step for {model_type}: {e}", exc_info=True)
            raise

        # --- Logging ---
        loss_val = loss.item()
        if batch_idx % log_interval == 0:
            log_msg = (
                f"Train Epoch: {epoch+1} [{batch_idx * len(state)}/{len(self.train_loader.dataset)} "
                f"({100. * batch_idx / len(self.train_loader):.0f}%)]\t{model_type.capitalize()} Loss: {loss_val:.6f}"
            )
            logger.info(log_msg)
            if self.wandb_run:
                log_data = {f"Train/{model_type}_batch_loss": loss_val}
                # Log components of VICRegLoss if applicable
                if isinstance(loss_info, VICRegLossInfo) and isinstance(criterion, VICRegLoss):
                     log_data[f"Train/{model_type}_sim_loss"] = loss_info.sim_loss
                     log_data[f"Train/{model_type}_std_loss"] = loss_info.std_loss
                     log_data[f"Train/{model_type}_cov_loss"] = loss_info.cov_loss
                self.wandb_run.log(log_data) # Log batch loss to wandb

        return loss_val


    def train_dynamics(self, epochs=None, log_interval=10):
        """
        Train the dynamics model for the specified number of epochs.
        
        Args:
            epochs: Number of epochs to train for (defaults to config value).
            log_interval: How often to log training metrics.
        
        Returns:
            Dictionary of training metrics.
        """
        epochs = epochs if epochs is not None else self.config.get("training", {}).get("dynamics_epochs", 10)
        logger.info(f"Training dynamics model for {epochs} epochs...")

        if self.dynamics_model is None or self.dynamics_optimizer is None or self.dynamics_criterion is None:
            logger.error("Dynamics model, optimizer, or criterion not initialized. Cannot train.")
            return {"dynamics_train_loss": float('inf'), "dynamics_val_loss": float('inf')}

        if self.train_loader is None or self.val_loader is None:
             logger.error("Train or validation loader not available. Cannot train dynamics model.")
             return {"dynamics_train_loss": float('inf'), "dynamics_val_loss": float('inf')}

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        metrics = {}

        # Track when training starts for benchmarking
        start_time = time.time()

        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.dynamics_model.train() # Set model to training mode
            # Also set CNN encoder to train mode if it exists
            if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None:
                 self.cnn_encoder.train()

            total_train_loss = 0
            num_batches = 0

            # Wrap data loader with tqdm for progress bar
            train_iterator = tqdm(self.train_loader, desc=f"Dynamics Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, batch in enumerate(train_iterator):
                batch_loss = self.train_batch('dynamics', batch, epoch, batch_idx, log_interval)
                total_train_loss += batch_loss
                num_batches += 1
                # Update tqdm description if needed
                # train_iterator.set_postfix(loss=f"{batch_loss:.4f}")

            avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_train_loss)

            # Validation phase
            avg_val_loss = self.validate_dynamics()
            val_losses.append(avg_val_loss)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            logger.info(
                f"Dynamics Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_duration:.2f}s"
            )

            # WandB logging for epoch
            if self.wandb_run:
                log_epoch_data = {
                    "Epoch": epoch + 1,
                    "Dynamics/Train_Loss_Epoch": avg_train_loss,
                    "Dynamics/Val_Loss_Epoch": avg_val_loss,
                    "Dynamics/Epoch_Time": epoch_duration,
                    "Dynamics/LR": self.dynamics_optimizer.param_groups[0]['lr'] # Log learning rate
                }
                self.wandb_run.log(log_epoch_data)

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"New best dynamics validation loss: {best_val_loss:.4f}. Saving model...")
                self.save_model('dynamics') # Saves dynamics_model.pt and potentially cnn_encoder.pt

                # WandB save best metric
                if self.wandb_run:
                     self.wandb_run.summary["best_dynamics_val_loss"] = best_val_loss

        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(f"Finished training dynamics model. Total time: {total_duration:.2f}s")

        metrics = {
            "dynamics_train_losses": train_losses,
            "dynamics_val_losses": val_losses,
            "best_dynamics_val_loss": best_val_loss,
            "total_dynamics_train_time": total_duration
        }
        return metrics


    def train_reward(self, epochs=None, log_interval=10):
        """
        Train the reward model for the specified number of epochs.
        
        Args:
            epochs: Number of epochs to train for (defaults to config value).
            log_interval: How often to log training metrics.
        
        Returns:
            Dictionary of training metrics.
        """
        epochs = epochs if epochs is not None else self.config.get("training", {}).get("reward_epochs", 10)
        logger.info(f"Training reward model for {epochs} epochs...")

        if self.reward_model is None or self.reward_optimizer is None or self.reward_criterion is None:
            logger.error("Reward model, optimizer, or criterion not initialized. Cannot train.")
            return {"reward_train_loss": float('inf'), "reward_val_loss": float('inf')}

        if self.train_loader is None or self.val_loader is None:
             logger.error("Train or validation loader not available. Cannot train reward model.")
             return {"reward_train_loss": float('inf'), "reward_val_loss": float('inf')}

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        metrics = {}

        start_time = time.time()

        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.reward_model.train() # Set model to training mode
            # Note: CNN encoder is typically NOT trained with reward model, kept frozen.

            total_train_loss = 0
            num_batches = 0

            train_iterator = tqdm(self.train_loader, desc=f"Reward Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, batch in enumerate(train_iterator):
                batch_loss = self.train_batch('reward', batch, epoch, batch_idx, log_interval)
                total_train_loss += batch_loss
                num_batches += 1

            avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_train_loss)

            # Validation phase
            avg_val_loss = self.validate_reward()
            val_losses.append(avg_val_loss)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            logger.info(
                f"Reward Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_duration:.2f}s"
            )

            # WandB logging
            if self.wandb_run:
                log_epoch_data = {
                    "Epoch": epoch + 1,
                    "Reward/Train_Loss_Epoch": avg_train_loss,
                    "Reward/Val_Loss_Epoch": avg_val_loss,
                    "Reward/Epoch_Time": epoch_duration,
                    "Reward/LR": self.reward_optimizer.param_groups[0]['lr']
                }
                self.wandb_run.log(log_epoch_data)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"New best reward validation loss: {best_val_loss:.4f}. Saving model...")
                self.save_model('reward')

                if self.wandb_run:
                     self.wandb_run.summary["best_reward_val_loss"] = best_val_loss


        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(f"Finished training reward model. Total time: {total_duration:.2f}s")

        metrics = {
            "reward_train_losses": train_losses,
            "reward_val_losses": val_losses,
            "best_reward_val_loss": best_val_loss,
            "total_reward_train_time": total_duration
        }
        return metrics


    def train_all(self):
        """
        Run the full training process for enabled models.
        
        Returns:
            Dictionary containing training metrics for dynamics and reward models.
        """
        all_metrics = {}
        logger.info("Starting PLDM training process...")

        if self.config.get("training", {}).get("train_dynamics", True):
            if self.dynamics_model:
                logger.info("--- Training Dynamics Model ---")
                dynamics_metrics = self.train_dynamics()
                all_metrics.update(dynamics_metrics)
            else:
                logger.warning("Dynamics training enabled but model not initialized. Skipping.")
        else:
            logger.info("Dynamics model training is disabled by configuration.")

        if self.config.get("training", {}).get("train_reward", True):
            if self.reward_model:
                logger.info("--- Training Reward Model ---")
                reward_metrics = self.train_reward()
                all_metrics.update(reward_metrics)
            else:
                logger.warning("Reward training enabled but model not initialized. Skipping.")
        else:
            logger.info("Reward model training is disabled by configuration.")

        logger.info("PLDM training process finished.")

        # Optionally evaluate final models on test set here if needed

        return all_metrics


    def validate_dynamics(self):
        """
        Validate the dynamics model on the validation set.
        
        Returns:
            Average validation loss (float).
        """
        if self.dynamics_model is None or self.val_loader is None:
            logger.warning("Cannot validate dynamics: Model or validation loader not initialized.")
            return float('inf')

        self.dynamics_model.eval()
        # Also set CNN encoder to eval mode if it exists
        if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None:
            self.cnn_encoder.eval()

        total_loss = 0
        total_batches = 0
        criterion = MSELoss() # Use simple MSE for validation reporting consistency

        with torch.no_grad():
            val_iterator = tqdm(self.val_loader, desc="Validating Dynamics", leave=False)
            for batch in val_iterator:
                state, action, next_state, _ = batch
                state = state.to(self.device)
                action = action.to(self.device)
                next_state = next_state.to(self.device)

                # Handle state encoding
                if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None and self.model_type == 'grid':
                    state_repr = self.cnn_encoder(state)
                    next_state_repr = self.cnn_encoder(next_state)
                else:
                    state_repr = state
                    next_state_repr = next_state

                # Get prediction
                # Handle different model signatures (e.g., teacher forcing off during eval)
                if isinstance(self.dynamics_model, TransformerPredictor):
                     # No target_seq or teacher forcing during validation
                     pred_next_state_repr = self.dynamics_model(state_repr, action)
                else:
                     pred_next_state_repr = self.dynamics_model(state_repr, action)

                # Calculate loss (MSE between predicted and actual next state representation)
                loss_info = criterion(pred_next_state_repr, next_state_repr)
                loss = loss_info.total_loss

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        return avg_loss


    def validate_reward(self):
        """
        Validate the reward model on the validation set.
        
        Returns:
            Average validation loss (float).
        """
        if self.reward_model is None or self.val_loader is None:
            logger.warning("Cannot validate reward: Model or validation loader not initialized.")
            return float('inf')

        self.reward_model.eval()
        # CNN encoder is usually frozen during reward validation/training
        if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None:
            self.cnn_encoder.eval() # Keep in eval mode

        total_loss = 0
        total_batches = 0
        criterion = MSELoss() # Use simple MSE for validation reporting

        with torch.no_grad():
            val_iterator = tqdm(self.val_loader, desc="Validating Reward", leave=False)
            for batch in val_iterator:
                state, action, _, reward = batch
                state = state.to(self.device)
                action = action.to(self.device)
                reward = reward.to(self.device).float().unsqueeze(1) # Target shape [batch_size, 1]

                # Handle state encoding
                if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None and self.model_type == 'grid':
                    state_repr = self.cnn_encoder(state)
                else:
                    state_repr = state

                # Get prediction
                pred_reward = self.reward_model(state_repr, action)

                # Calculate loss (MSE between predicted and actual reward)
                loss_info = criterion(pred_reward, reward)
                loss = loss_info.total_loss

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        return avg_loss


    def save_model(self, model_name):
        """
        Save model to disk. Saves both predictor and CNN encoder if used.
        
        Args:
            model_name: Name of the model to save ('dynamics' or 'reward')
        """
        model_to_save = None
        optimizer_to_save = None
        if model_name == 'dynamics':
            model_to_save = self.dynamics_model
            optimizer_to_save = self.dynamics_optimizer
        elif model_name == 'reward':
            model_to_save = self.reward_model
            optimizer_to_save = self.reward_optimizer
        else:
            logger.error(f"Unknown model name for saving: {model_name}")
            return

        if model_to_save is None:
            logger.error(f"Cannot save {model_name} model: Model not initialized")
            return

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Save Predictor Model ---
        predictor_path = os.path.join(self.output_dir, f"{model_name}_model.pt")
        predictor_state = {
            'model_state_dict': model_to_save.state_dict(),
            'config': self.config, # Save the full config used for training
            # Optionally save optimizer state if needed for resuming training
             'optimizer_state_dict': optimizer_to_save.state_dict() if optimizer_to_save else None,
        }
        try:
            torch.save(predictor_state, predictor_path)
            logger.info(f"Saved {model_name} predictor model to {predictor_path}")
            # Log artifact to WandB if enabled
            if self.wandb_run and self.config.get("wandb", {}).get("save_model", True):
                 artifact = wandb.Artifact(f'{model_name}-predictor', type='model')
                 artifact.add_file(predictor_path)
                 self.wandb_run.log_artifact(artifact)
                 logger.info(f"Logged {model_name} predictor artifact to WandB")

        except Exception as e:
            logger.error(f"Error saving {model_name} predictor model: {e}", exc_info=True)


        # --- Save CNN Encoder (if exists and associated with this model type) ---
        # Conventionally, CNN encoder is trained/saved with dynamics model
        if model_name == 'dynamics' and hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None:
            cnn_encoder_path = os.path.join(self.output_dir, "cnn_encoder.pt")
            cnn_encoder_state = {
                'model_state_dict': self.cnn_encoder.state_dict(),
                'config': self.config, # Save config as well
                 # Optimizer state for CNN params is within dynamics_optimizer, saved above
            }
            try:
                torch.save(cnn_encoder_state, cnn_encoder_path)
                logger.info(f"Saved CNN encoder model to {cnn_encoder_path}")
                # Log artifact to WandB
                if self.wandb_run and self.config.get("wandb", {}).get("save_model", True):
                     artifact = wandb.Artifact('cnn-encoder', type='model')
                     artifact.add_file(cnn_encoder_path)
                     self.wandb_run.log_artifact(artifact)
                     logger.info("Logged CNN encoder artifact to WandB")
            except Exception as e:
                logger.error(f"Error saving CNN encoder model: {e}", exc_info=True)


    def load_model(self, model_name, model_path=None, cnn_encoder_path=None):
        """
        Load model from disk. Loads both predictor and CNN encoder if paths provided.
        NOTE: Loading is currently disabled as requested.
        
        Args:
            model_name: Name of the model to load ('dynamics' or 'reward').
            model_path: Path to the predictor model file (.pt). Defaults to standard path in output_dir.
            cnn_encoder_path: Path to the CNN encoder file (.pt). Defaults to standard path. Required if model uses CNN.
        """
        logger.info(f"Skipping load_model for '{model_name}' as requested. Models will be trained from scratch.")
        return # Exit the function immediately, preventing any loading
        
        # --- Original loading code below (commented out or removed) ---
        # # Determine default paths if not provided
        # if model_path is None:
        #     model_path = os.path.join(self.output_dir, f"{model_name}_model.pt")
        # if cnn_encoder_path is None and model_name == 'dynamics': # Only default load CNN for dynamics
        #      cnn_encoder_path = os.path.join(self.output_dir, "cnn_encoder.pt")
        # 
        # # --- Load Predictor Model ---
        # if os.path.exists(model_path):
        #     try:
        #         checkpoint = torch.load(model_path, map_location=self.device)
        #         
        #         # Handle different checkpoint formats
        #         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        #             model_state_dict = checkpoint['model_state_dict']
        #             optimizer_state_dict = checkpoint.get('optimizer_state_dict')
        #         else:
        #             # Assume the checkpoint is the state_dict itself (older format)
        #             model_state_dict = checkpoint
        #             optimizer_state_dict = None
        #             logger.warning(f"Loaded checkpoint for {model_name} seems to be only a state_dict. Optimizer state not loaded.")
        #         
        #         if model_name == 'dynamics' and self.dynamics_model:
        #             self.dynamics_model.load_state_dict(model_state_dict)
        #             # Load optimizer state if available and needed
        #             if optimizer_state_dict and self.dynamics_optimizer:
        #                  try:
        #                      self.dynamics_optimizer.load_state_dict(optimizer_state_dict)
        #                  except Exception as opt_e:
        #                       logger.warning(f"Could not load dynamics optimizer state from checkpoint: {opt_e}")
        #             logger.info(f"Loaded {model_name} predictor model from {model_path}")
        #         elif model_name == 'reward' and self.reward_model:
        #             self.reward_model.load_state_dict(model_state_dict)
        #             if optimizer_state_dict and self.reward_optimizer:
        #                 try:
        #                      self.reward_optimizer.load_state_dict(optimizer_state_dict)
        #                 except Exception as opt_e:
        #                       logger.warning(f"Could not load reward optimizer state from checkpoint: {opt_e}")
        #             logger.info(f"Loaded {model_name} predictor model from {model_path}")
        #         else:
        #             logger.warning(f"{model_name.capitalize()} model not initialized, cannot load state dict from {model_path}")
        # 
        #     except FileNotFoundError:
        #         logger.error(f"Predictor model file not found at {model_path}. Cannot load {model_name} model.")
        #     except Exception as e:
        #         logger.error(f"Error loading {model_name} predictor model from {model_path}: {e}", exc_info=True)
        # else:
        #      logger.warning(f"Predictor model file {model_path} does not exist. Model not loaded.")
        # 
        # 
        # # --- Load CNN Encoder ---
        # # Only load if cnn_encoder exists on the trainer AND path is valid AND it's for dynamics
        # if model_name == 'dynamics' and hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None:
        #     if cnn_encoder_path and os.path.exists(cnn_encoder_path):
        #         try:
        #             cnn_checkpoint = torch.load(cnn_encoder_path, map_location=self.device)
        #             
        #             # Handle different checkpoint formats
        #             if isinstance(cnn_checkpoint, dict) and 'model_state_dict' in cnn_checkpoint:
        #                 cnn_model_state_dict = cnn_checkpoint['model_state_dict']
        #             else:
        #                 cnn_model_state_dict = cnn_checkpoint
        #                 logger.warning("Loaded CNN checkpoint seems to be only a state_dict.")
        #                 
        #             self.cnn_encoder.load_state_dict(cnn_model_state_dict)
        #             logger.info(f"Loaded CNN encoder from {cnn_encoder_path}")
        #             # Note: CNN optimizer state is part of dynamics_optimizer state, loaded above.
        #         except FileNotFoundError:
        #              logger.error(f"CNN encoder file not found at {cnn_encoder_path}. Cannot load.")
        #         except Exception as e:
        #             logger.error(f"Error loading CNN encoder model from {cnn_encoder_path}: {e}", exc_info=True)
        #     else:
        #          logger.warning(f"CNN encoder path {cnn_encoder_path} not found or not specified. CNN encoder not loaded.")


# Example usage (within another script like train_pldm.py)
if __name__ == '__main__':
    # This is placeholder example code
    logging.basicConfig(level=logging.INFO)
    logger.info("PLDM Trainer module direct execution (example usage)")

    # Dummy config
    dummy_config = {
        "seed": 42,
        "model": {"type": "grid", "encoder_type": "cnn", "state_embed_dim": 64,
                  "dynamics_predictor": {"predictor_type": "transformer"},
                  "reward_predictor": {"predictor_type": "grid"}
                  },
        "data": {"train_data_path": "dummy_train.csv", "grid_size": 5},
        "training": {"batch_size": 4, "dynamics_epochs": 1, "reward_epochs": 1,
                     "train_dynamics": True, "train_reward": True},
        "loss": {"dynamics_loss": "mse", "reward_loss": "mse"},
        "wandb": {"use_wandb": False}
    }

    # Need dummy data loaders for initialization
    # Replace with actual data loading
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 16
        def __getitem__(self, idx):
            # B, C, H, W
            state = torch.randn(32, 5, 5)
            action = torch.randint(0, 6, (1,)).squeeze() # scalar action
            next_state = torch.randn(32, 5, 5)
            reward = torch.randn(1).squeeze()
            return state, action, next_state, reward

    dummy_loader = DataLoader(DummyDataset(), batch_size=4)

    try:
        trainer = PLDMTrainer(
            config=dummy_config,
            output_dir="temp_pldm_output",
            train_loader=dummy_loader, # Provide dummy loaders
            val_loader=dummy_loader
        )
        logger.info("Dummy Trainer initialized.")
        # trainer.train_all() # Run training
        # trainer.save_model('dynamics')
        # trainer.save_model('reward')
        # trainer.load_model('dynamics')
        logger.info("Dummy run completed.")

    except Exception as e:
        logger.error(f"Error during dummy trainer execution: {e}", exc_info=True)

    # Clean up dummy files/dirs if needed
    # import shutil
    # if os.path.exists("temp_pldm_output"):
    #     shutil.rmtree("temp_pldm_output") 