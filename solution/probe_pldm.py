import argparse
import os
import sys
import json
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
from dataclasses import dataclass
import torch.nn.functional as F
import gc  # Import garbage collection
import inspect

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from solution.pldm.trainer import PLDMTrainer
from solution.pldm.config import load_config, merge_configs, get_default_config
from solution.pldm.utils import setup_logger, set_seeds
from solution.pldm.prober import Prober, VectorProber
from solution.pldm.data_processor import get_overcooked_dataloaders

# Attempt to import wandb, but don't fail if it's not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Get logger for this module
logger = logging.getLogger(__name__)


@dataclass
class ProbeTargetConfig:
    """Configuration for a specific probe target."""
    arch: Optional[str] = "mlp"
    subclass: Optional[str] = None
    hidden_dims: Tuple[int, ...] = (128, 64)


@dataclass
class ProbingConfig:
    """Main configuration for the probing process."""
    # Target configuration
    state_channels: ProbeTargetConfig = ProbeTargetConfig()
    agent_pos: ProbeTargetConfig = ProbeTargetConfig()
    reward: ProbeTargetConfig = ProbeTargetConfig()
    
    # Training parameters
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 64
    seed: int = 42
    max_samples: Optional[int] = None
    
    # Model selection
    probe_dynamics: bool = True
    probe_reward: bool = True
    probe_encoder: bool = True    # Probe state encoder representations
    probe_predictor: bool = False  # Probe dynamics predictor representations
    
    # Visualization
    visualize: bool = True
    max_vis_samples: int = 10
    
    # Save options
    save_weights: bool = True     # Whether to save prober weights to disk
    save_visualizations: bool = True  # Whether to generate and save visualizations
    
    # Other settings
    use_wandb: bool = False
    full_finetune: bool = False


class ProbeResult(NamedTuple):
    """Stores results from a probing task."""
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_channel: List[float]
    plots: List[Any]


class StateChannels:
    """Enumeration of different state channels that can be probed."""
    AGENT1_POS = 0
    AGENT1_ORIENTATION_UP = 1
    AGENT1_ORIENTATION_DOWN = 2
    AGENT1_ORIENTATION_LEFT = 3
    AGENT1_ORIENTATION_RIGHT = 4
    AGENT2_POS = 5
    AGENT2_ORIENTATION_UP = 6
    AGENT2_ORIENTATION_DOWN = 7
    AGENT2_ORIENTATION_LEFT = 8
    AGENT2_ORIENTATION_RIGHT = 9
    AGENT1_HOLDING_ONION = 10
    AGENT1_HOLDING_TOMATO = 11
    AGENT1_HOLDING_DISH = 12
    AGENT1_HOLDING_SOUP = 13
    AGENT2_HOLDING_ONION = 14
    AGENT2_HOLDING_TOMATO = 15
    AGENT2_HOLDING_DISH = 16
    AGENT2_HOLDING_SOUP = 17
    ONION_ON_COUNTER = 18
    TOMATO_ON_COUNTER = 19
    DISH_ON_COUNTER = 20
    SOUP_ON_COUNTER = 21
    POT_EMPTY = 22
    POT_COOKING = 23
    POT_READY = 24
    WALL = 25
    COUNTER = 26
    ONION_DISPENSER = 27
    TOMATO_DISPENSER = 28
    DISH_DISPENSER = 29
    SERVING_LOCATION = 30
    TIMESTEP = 31
    
    @classmethod
    def get_channel_names(cls):
        return [name for name in dir(cls) if not name.startswith('__') and not callable(getattr(cls, name))]
    
    @classmethod
    def get_channel_values(cls):
        return [getattr(cls, name) for name in cls.get_channel_names()]
    
    @classmethod
    def get_channel_dict(cls):
        return {name: getattr(cls, name) for name in cls.get_channel_names()}


def squared_error_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute squared error loss between prediction and target.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        
    Returns:
        Per-element squared error
    """
    assert pred.shape == target.shape
    return (pred - target).pow(2)


class ProbingEvaluator:
    """Evaluates representations from PLDM models by training probers."""
    
    def __init__(
        self,
        model: PLDMTrainer,
        output_dir: str,
        config: ProbingConfig,
        device: torch.device = None,
        wandb_run = None,
    ):
        """
        Initialize the ProbingEvaluator.
        
        Args:
            model: Trained PLDMTrainer containing models to probe
            output_dir: Directory to save probing results
            config: Probing configuration
            device: Device to use
            wandb_run: Optional wandb run for logging
        """
        self.model = model
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.config = config
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Wandb setup
        self.wandb_run = wandb_run
        self.use_wandb = WANDB_AVAILABLE and (self.wandb_run is not None or config.use_wandb)
        
        # Set random seed
        self.seed = config.seed
        self._set_seed()
        
        # Data loaders for training and evaluation
        self.train_loader = model.train_loader
        self.val_loader = model.val_loader
        self.test_loader = model.test_loader
        
        # Store model dimensions
        self.state_dims = self._get_state_dimensions()
        self.model_type = model.model_type
        
        # Dictionary to store probers
        self.probers = {}
        
        logger.info(f"ProbingEvaluator initialized with {self.model_type} model")
    
    def _set_seed(self):
        """Set random seed for reproducibility."""
        set_seeds(self.seed)
        logger.debug(f"Seed set to {self.seed}")
        
    def _get_state_dimensions(self):
        """Get dimensions of the state representation."""
        if not self.train_loader:
            logger.error("No train loader available to determine state dimensions")
            return None
            
        sample_batch = next(iter(self.train_loader))
        state, _, _, _ = sample_batch
        
        return {
            "batch_size": state.size(0),
            "channels": state.size(1) if len(state.shape) > 3 else 1,
            "height": state.size(2) if len(state.shape) > 3 else state.size(1),
            "width": state.size(3) if len(state.shape) > 3 else state.size(2)
        }
        
    def _context_manager(self):
        """Context manager for gradient computation based on whether we're finetuning."""
        return torch.enable_grad() if self.config.full_finetune else torch.no_grad()
        
    def _get_model_representation(self, batch, model_type, repr_type='encoder'):
        """
        Extract representation from the model.
        
        Args:
            batch: Input batch of data
            model_type: 'dynamics' or 'reward'
            repr_type: 'encoder' or 'predictor'
            
        Returns:
            Model representation
        """
        state, action, next_state, reward = batch
        
        # Move to device
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device) # Ensure next_state is also on device
        
        # Get the appropriate model
        if model_type == 'dynamics':
            model = self.model.dynamics_model
        else:  # reward
            model = self.model.reward_model
        
        # Check if CNN encoder exists and should be used
        # Assumes self.model is the PLDMTrainer instance or has similar attributes
        use_cnn = hasattr(self.model, 'cnn_encoder') and self.model.cnn_encoder is not None
        
        # Extract representation based on type
        with torch.no_grad():
            if use_cnn and self.model_type == 'grid':
                # Use CNN encoder first
                state_embed = self.model.cnn_encoder(state)
                next_state_embed = self.model.cnn_encoder(next_state)
            else:
                # Use raw state if no CNN or not grid model
                state_embed = state
                next_state_embed = next_state
                
            # Now use the embeddings (or raw states) with the predictor/encoder logic
            if repr_type == 'encoder':
                # For encoder probing, we want the representation *before* the predictor
                # If CNN was used, state_embed is the CNN output. Otherwise, it's raw state.
                # If the model itself is just an encoder (e.g., CNNStateEncoderNetwork passed directly), 
                # we might need different logic, but here we assume model is Dynamics/Reward Predictor.
                # Let's return state_embed which is either raw state or CNN output
                representation = state_embed
            else:  # predictor
                # For predictor probing, we want the output of the dynamics/reward predictor
                # The predictor takes the state embedding (or raw state) and action
                
                # Check if model is Transformer or similar that needs return_embedding=True for predictor's internal rep
                from solution.pldm.predictors import TransformerPredictor, LSTMPredictor
                if isinstance(model, (TransformerPredictor, LSTMPredictor)):
                     # These models predict the *next* state embedding when return_embedding=True
                    representation = model(state_embed, action, return_embedding=True)
                elif hasattr(model, 'forward') and 'return_embedding' in inspect.signature(model.forward).parameters:
                     # Other predictors might use return_embedding for their internal state
                    representation = model(state_embed, action, return_embedding=True)
                else:
                     # Fallback: use the standard output of the model as the representation
                    # For dynamics, this might be the predicted next state grid/vector (not embedding)
                    # For reward, this would be the scalar reward
                    representation = model(state_embed, action)
                
        return representation

    def _get_state_channel_target(self, batch, channel_index, repr_type='encoder'):
        """
        Extract a specific channel from the state tensor as a probing target.
        
        Args:
            batch: Input batch
            channel_index: Index of channel to extract
            repr_type: The representation type being used ('encoder' or 'predictor')
            
        Returns:
            Channel data as target tensor
        """
        # For predictor representation, we want to predict the next state channels
        # For encoder representation, we use the current state channels
        if repr_type == 'predictor':
            _, _, state_tensor, _ = batch  # Use next_state for predictor
        else:
            state_tensor, _, _, _ = batch  # Use current state for encoder
        
        state_tensor = state_tensor.to(self.device)
        
        if self.model_type == "grid" and len(state_tensor.shape) > 3:
            # For grid states, extract the specific channel
            target = state_tensor[:, channel_index, :, :]
            # Flatten spatial dimensions
            target = target.reshape(target.size(0), -1)
        else:
            # For vector states, this is harder - assume all data is flattened
            raise ValueError("State channel extraction not supported for vector states")
            
        return target
            
    def create_prober(self, input_dim, output_dim, config: ProbeTargetConfig):
        """
        Create a prober model with the specified dimensions.
        
        Args:
            input_dim: Dimension of input representation
            output_dim: Dimension of output to predict
            config: Configuration for this prober
            
        Returns:
            Initialized prober model
        """
        logger.info(f"Creating prober with input_dim={input_dim}, output_dim={output_dim}")
        
        if config.arch == "mlp":
            return VectorProber(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=config.hidden_dims
            )
        else:
            raise ValueError(f"Unsupported prober architecture: {config.arch}")
            
    def train_prober_for_channel(
        self,
        channel_index: int,
        model_type: str = 'dynamics',
        repr_type: str = 'encoder',
        epoch: int = 0
    ):
        """
        Train a prober to predict a specific state channel from model representations.
        
        Args:
            channel_index: Index of state channel to probe
            model_type: 'dynamics' or 'reward'
            repr_type: 'encoder' or 'predictor'
            epoch: Current epoch number
            
        Returns:
            Trained prober model
        """
        # Set seed for reproducibility
        self._set_seed()
        channel_names = StateChannels.get_channel_names()
        channel_name = channel_names[channel_index] if channel_index < len(channel_names) else f"channel_{channel_index}"
        logger.info(f"Training {model_type} {repr_type} prober for channel: {channel_name} (index {channel_index})")
        
        # Get data loaders
        train_loader = self.train_loader
        val_loader = self.val_loader
        
        if not train_loader or not val_loader:
            raise ValueError("Train or validation loader not available")
        
        # Get a batch to determine dimensions
        batch = next(iter(train_loader))
        representation = self._get_model_representation(batch, model_type, repr_type)
        target = self._get_state_channel_target(batch, channel_index, repr_type)
        
        input_dim = representation.size(1)
        output_dim = target.size(1)
        
        # Create prober
        self._set_seed()  # Reset seed for consistent initialization
        config = self.config.state_channels
        prober = self.create_prober(input_dim, output_dim, config)
        prober = prober.to(self.device)
        
        # Set up optimizer
        optimizer = torch.optim.Adam(prober.parameters(), lr=self.config.lr)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training phase
            prober.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} - Training"):
                representation = self._get_model_representation(batch, model_type, repr_type)
                target = self._get_state_channel_target(batch, channel_index, repr_type)
                
                optimizer.zero_grad()
                prediction = prober(representation)
                loss = criterion(prediction, target)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            
            # Validation phase
            prober.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} - Validation"):
                    representation = self._get_model_representation(batch, model_type, repr_type)
                    target = self._get_state_channel_target(batch, channel_index, repr_type)
                    
                    prediction = prober(representation)
                    loss = criterion(prediction, target)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Log to WandB
            if self.use_wandb:
                self.wandb_run.log({
                    f"prober_{model_type}_{repr_type}_{channel_name}/train_loss": avg_train_loss,
                    f"prober_{model_type}_{repr_type}_{channel_name}/val_loss": avg_val_loss,
                    f"prober_{model_type}_{repr_type}_{channel_name}/epoch": epoch + 1
                })
            
            # Save best model if configured to do so
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
                # Save model weights if configured
                if self.config.save_weights:
                    prober_path = self.output_dir / f"prober_{model_type}_{repr_type}_{channel_name}.pt"
                
                    torch.save({
                        'state_dict': prober.state_dict(),
                        'config': {
                            'input_dim': input_dim,
                            'output_dim': output_dim,
                            'model_type': model_type,
                            'repr_type': repr_type,
                            'channel_index': channel_index,
                            'channel_name': channel_name
                        }
                    }, prober_path)
                
                    logger.info(f"New best {model_type} {repr_type} prober for {channel_name} saved to {prober_path}")
                else:
                    logger.info(f"New best {model_type} {repr_type} prober for {channel_name} (weights not saved)")
        
        # Store the prober
        key = f"{model_type}_{repr_type}_{channel_name}"
        self.probers[key] = prober
        
        return prober
    
    def evaluate_prober_for_channel(
        self,
        prober,
        channel_index: int,
        model_type: str = 'dynamics',
        repr_type: str = 'encoder',
        visualize: bool = True
    ):
        """
        Evaluate a trained prober on the test set.
        
        Args:
            prober: Trained prober model
            channel_index: Index of state channel being probed
            model_type: 'dynamics' or 'reward'
            repr_type: 'encoder' or 'predictor'
            visualize: Whether to create visualizations
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Set seed for reproducibility
        self._set_seed()
        channel_names = StateChannels.get_channel_names()
        channel_name = channel_names[channel_index] if channel_index < len(channel_names) else f"channel_{channel_index}"
        logger.info(f"Evaluating {model_type} {repr_type} prober for channel: {channel_name}")
        
        if not self.test_loader:
                logger.error("No test loader available for evaluation")
                return {}
            
        test_loader = self.test_loader
        
        prober.eval()
        criterion = torch.nn.MSELoss(reduction='none')
        
        all_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {channel_name} prober"):
                representation = self._get_model_representation(batch, model_type, repr_type)
                target = self._get_state_channel_target(batch, channel_index, repr_type)
                
                prediction = prober(representation)
                loss = criterion(prediction, target)
                
                all_losses.append(loss)
                all_predictions.append(prediction.cpu())
                all_targets.append(target.cpu())
        
        # Concatenate results
        all_losses = torch.cat(all_losses, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        mean_loss = all_losses.mean().item()
        rmse = torch.sqrt(all_losses.mean()).item()
        
        # Calculate per-element losses to understand which parts are predicted well
        per_element_loss = all_losses.mean(dim=0)
        
        # R²-score calculation
        target_var = torch.var(all_targets, dim=0, unbiased=False)
        unexplained_var = torch.var(all_targets - all_predictions, dim=0, unbiased=False)
        r2_score = 1 - (unexplained_var / target_var)
        mean_r2 = r2_score.mean().item()
        
        # Compile metrics
        metrics = {
            'mean_loss': mean_loss,
            'rmse': rmse,
            'r2_score': mean_r2,
            'per_element_loss': per_element_loss.tolist()
        }
        
        logger.info(f"Prober evaluation results for {model_type}_{repr_type}_{channel_name}:")
        logger.info(f"  Mean Loss: {mean_loss:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  R² Score: {mean_r2:.6f}")
        
        # Log to WandB
        if self.use_wandb:
            self.wandb_run.log({
                f"prober_{model_type}_{repr_type}_{channel_name}/test_loss": mean_loss,
                f"prober_{model_type}_{repr_type}_{channel_name}/test_rmse": rmse,
                f"prober_{model_type}_{repr_type}_{channel_name}/test_r2": mean_r2
            })
        
        # Visualize results if requested
        if visualize and self.config.visualize and self.config.save_visualizations:
            self.visualize_channel_results(
                predictions=all_predictions,
                targets=all_targets,
                channel_index=channel_index,
                model_type=model_type,
                repr_type=repr_type
            )
        elif visualize and self.config.visualize:
            logger.info(f"Skipping visualization for {model_type}_{repr_type}_{channel_name} (save_visualizations disabled)")
        
        return metrics
    
    def visualize_channel_results(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        channel_index: int,
        model_type: str,
        repr_type: str
    ):
        """
        Visualize the predictions vs targets for a specific channel.
        
        Args:
            predictions: Tensor of predictions
            targets: Tensor of ground truth values
            channel_index: Index of state channel being visualized
            model_type: 'dynamics' or 'reward'
            repr_type: 'encoder' or 'predictor'
        """
        channel_names = StateChannels.get_channel_names()
        channel_name = channel_names[channel_index] if channel_index < len(channel_names) else f"channel_{channel_index}"
        
        # Determine number of samples to visualize
        n_samples = min(self.config.max_vis_samples, predictions.size(0))
        n_elements = min(100, predictions.size(1))  # Limit elements for clarity in visualization
        
        # Convert to numpy for plotting
        predictions_np = predictions[:n_samples, :n_elements].numpy()
        targets_np = targets[:n_samples, :n_elements].numpy()
        
        # Calculate grid dimensions
        h = w = int(np.sqrt(n_elements))
        if h * w < n_elements:
            w += 1
            if h * w < n_elements:
                h += 1
                
        # Create heatmap visualizations for select samples
        for i in range(n_samples):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot target
            target_reshaped = targets_np[i].reshape(h, w)
            im1 = ax1.imshow(target_reshaped, cmap='viridis')
            ax1.set_title('Target')
            plt.colorbar(im1, ax=ax1)
            
            # Plot prediction
            pred_reshaped = predictions_np[i].reshape(h, w)
            im2 = ax2.imshow(pred_reshaped, cmap='viridis')
            ax2.set_title('Prediction')
            plt.colorbar(im2, ax=ax2)
            
            # Plot difference
            diff = np.abs(targets_np[i] - predictions_np[i]).reshape(h, w)
            im3 = ax3.imshow(diff, cmap='hot')
            ax3.set_title('Absolute Difference')
            plt.colorbar(im3, ax=ax3)
            
            plt.suptitle(f'{model_type.capitalize()} Model - {repr_type.capitalize()} {channel_name} Predictions (Sample {i+1})')
            
            # Save figure
            plt_path = self.output_dir / f"{model_type}_{repr_type}_{channel_name}_sample{i+1}.png"
            plt.savefig(plt_path)
            
            # Log to WandB
            if self.use_wandb:
                self.wandb_run.log({f"{model_type}_{repr_type}_{channel_name}_sample{i+1}": wandb.Image(plt)})
                
            plt.close(fig)
            
        # Create a correlation plot with all samples
        if n_elements <= 10:  # Only for small number of elements
            fig = plt.figure(figsize=(10, 8))
            
            # Flatten all samples
            all_preds = predictions_np.flatten()
            all_targets = targets_np.flatten()
            
            plt.scatter(all_targets, all_preds, alpha=0.3)
            
            # Add diagonal line for perfect predictions
            min_val = min(all_targets.min(), all_preds.min())
            max_val = max(all_targets.max(), all_preds.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{model_type.capitalize()} Model - {repr_type.capitalize()} {channel_name} Correlation')
            
            # Save figure
            plt_path = self.output_dir / f"{model_type}_{repr_type}_{channel_name}_correlation.png"
            plt.savefig(plt_path)
            
            # Log to WandB
            if self.use_wandb:
                self.wandb_run.log({f"{model_type}_{repr_type}_{channel_name}_correlation": wandb.Image(plt)})
                
            plt.close(fig)
    
    def train_probers_for_all_channels(self):
        """
        Train probers for all state channels with both dynamics and reward models.
        
        Returns:
            Dictionary of evaluation results
        """
        # Set seed for reproducibility
        self._set_seed()
        logger.info(f"Training probers for all state channels")
        
        results = {}
        
        # Get all channel indices to probe
        channel_indices = StateChannels.get_channel_values()
        channel_names = StateChannels.get_channel_names()
        
        # Define which models to probe
        models = []
        if self.config.probe_dynamics:
            models.append('dynamics')
        if self.config.probe_reward:
            models.append('reward')
            
        # Define which representation types to probe
        repr_types = []
        if self.config.probe_encoder:
            repr_types.append('encoder')
        if self.config.probe_predictor:
            repr_types.append('predictor')
            
        if not repr_types:
            logger.warning("No representation types selected for probing. Defaulting to encoder.")
            repr_types = ['encoder']
        
        # Save probing configuration for reproducibility
        probe_config_path = self.output_dir / "probe_config.json"
        with open(probe_config_path, 'w') as f:
            json.dump({
                'seed': self.seed,
                'model_type': self.model_type,
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs,
                'lr': self.config.lr,
                'channels': channel_names,
                'models': models,
                'repr_types': repr_types,
                'save_weights': self.config.save_weights,
                'save_visualizations': self.config.save_visualizations
            }, f, indent=2)
        logger.info(f"Saved probing configuration to {probe_config_path}")
        
        # Train and evaluate probers for each combination
        for model_type in models:
            for repr_type in repr_types:
                logger.info(f"Training probers for {model_type} model using {repr_type} representations")
                for idx, channel_index in enumerate(channel_indices):
                    channel_name = channel_names[idx]
                
                    # Set seed before each probing task
                    self._set_seed()
                    
                    # Train prober
                    prober = self.train_prober_for_channel(
                        channel_index=channel_index,
                        model_type=model_type,
                        repr_type=repr_type
                    )
                    
                    # Evaluate prober
                    metrics = self.evaluate_prober_for_channel(
                        prober=prober,
                        channel_index=channel_index,
                        model_type=model_type,
                        repr_type=repr_type
                    )
                
                    # Include seed in metrics for reference
                    metrics['seed'] = self.seed
                    results[(model_type, repr_type, channel_name)] = metrics
                    
                    # Free up memory after processing each channel
                    logger.info(f"Cleaning up memory after processing channel {channel_name}")
                    del prober
                    gc.collect()
                    torch.cuda.empty_cache()
                
        # Aggregate and summarize results
        self.aggregate_results(results)
        
        return results
    
    def aggregate_results(self, results: Dict):
        """
        Aggregate and summarize probing results.
        
        Args:
            results: Dict mapping (model_type, repr_type, channel_name) to metrics
        """
        # Create a summary table
        summary = []
        
        for (model_type, repr_type, channel_name), metrics in results.items():
            summary.append({
                'Model': model_type,
                'ReprType': repr_type,
                'Channel': channel_name,
                'MSE': metrics['mean_loss'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2_score']
            })
        
        # Sort by model type, representation type, then channel
        summary.sort(key=lambda x: (x['Model'], x['ReprType'], x['Channel']))
        
        # Print summary table
        logger.info("\n--- Probing Results Summary ---")
        logger.info(f"{'Model':<10} {'ReprType':<10} {'Channel':<25} {'MSE':<10} {'RMSE':<10} {'R²':<10}")
        logger.info("-" * 75)
        
        for row in summary:
            logger.info(
                f"{row['Model']:<10} {row['ReprType']:<10} {row['Channel']:<25} "
                f"{row['MSE']:<10.6f} {row['RMSE']:<10.6f} {row['R²']:<10.6f}"
            )
        
        # Save summary to JSON
        summary_path = self.output_dir / "probing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log summary to WandB
        if self.use_wandb:
            # Create a table
            wandb_table = wandb.Table(
                columns=["Model", "ReprType", "Channel", "MSE", "RMSE", "R²"]
            )
            
            for row in summary:
                wandb_table.add_data(
                    row['Model'],
                    row['ReprType'],
                    row['Channel'],
                    row['MSE'],
                    row['RMSE'],
                    row['R²']
                )
            
            self.wandb_run.log({"probing_summary": wandb_table})
            
        # Create comparative visualizations
        self.visualize_comparative_results(summary)
        
    def visualize_comparative_results(self, summary):
        """
        Create comparative visualizations across channels and models.
        
        Args:
            summary: List of result dictionaries
        """
        # Skip visualization if disabled
        if not self.config.save_visualizations:
            logger.info("Skipping comparative visualizations (save_visualizations disabled)")
            return
            
        # Group by model and representation type
        models = sorted(list(set(row['Model'] for row in summary)))
        repr_types = sorted(list(set(row['ReprType'] for row in summary)))
        
        num_models = len(models)
        num_repr_types = len(repr_types)
        
        # Create figure with subplots - 1 for each model and repr_type combination
        fig, axes = plt.subplots(num_models, num_repr_types, figsize=(6*num_repr_types, 8*num_models))
        # Handle single subplot case
        if num_models * num_repr_types == 1:
            axes = np.array([[axes]])
        elif num_models == 1:
            axes = np.array([axes])
        elif num_repr_types == 1:
            axes = np.array([[ax] for ax in axes])
        
        for i, model in enumerate(models):
            for j, repr_type in enumerate(repr_types):
                # Get results for this model and repr_type
                results = [row for row in summary if row['Model'] == model and row['ReprType'] == repr_type]
                
                if not results:
                    continue
                
                # Sort by R² score to see best-to-worst channels
                results.sort(key=lambda x: x['R²'], reverse=True)
                
                channels = [row['Channel'] for row in results]
                r2_scores = [row['R²'] for row in results]
                
                # Create horizontal bar chart
                y_pos = np.arange(len(channels))
                axes[i, j].barh(y_pos, r2_scores, align='center')
                axes[i, j].set_yticks(y_pos)
                axes[i, j].set_yticklabels(channels)
                axes[i, j].invert_yaxis()  # Labels read top-to-bottom
                axes[i, j].set_xlabel('R² Score')
                axes[i, j].set_title(f'{model.capitalize()} Model - {repr_type.capitalize()} Representations')
                
                # Add grid for readability
                axes[i, j].grid(axis='x', linestyle='--', alpha=0.7)
                
                # Add a line at R²=0
                axes[i, j].axvline(x=0, color='r', linestyle='-', alpha=0.3)
                
                # Make sure all axes have same x limits for fair comparison
                axes[i, j].set_xlim(min(-0.1, min(r2_scores) - 0.1), max(1.0, max(r2_scores) + 0.1))
                
        plt.tight_layout()
        plt_path = self.output_dir / "comparative_results.png"
        plt.savefig(plt_path)
        
        if self.use_wandb:
            self.wandb_run.log({"comparative_results": wandb.Image(plt)})
            
        plt.close(fig)
        
        # Also create a direct comparison between encoder and predictor for the same channels
        if len(repr_types) > 1:
            # Get all channels from summary
            all_channels = sorted(list(set(row['Channel'] for row in summary)))
            
            # For each model, compare encoder vs predictor performance
            for model in models:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Get encoder and predictor results for this model
                encoder_results = {row['Channel']: row['R²'] for row in summary if row['Model'] == model and row['ReprType'] == 'encoder'}
                predictor_results = {row['Channel']: row['R²'] for row in summary if row['Model'] == model and row['ReprType'] == 'predictor'}
                
                # Filter for channels that have both encoder and predictor results
                common_channels = sorted([ch for ch in all_channels if ch in encoder_results and ch in predictor_results])
                
                if not common_channels:
                    plt.close(fig)
                    continue
                
                encoder_scores = [encoder_results[ch] for ch in common_channels]
                predictor_scores = [predictor_results[ch] for ch in common_channels]
                
                # Plot a scatter of encoder vs predictor R² scores
                plt.scatter(encoder_scores, predictor_scores, alpha=0.7)
                
                # Add diagonal line for equal performance
                min_val = min(min(encoder_scores), min(predictor_scores)) - 0.1
                max_val = max(max(encoder_scores), max(predictor_scores)) + 0.1
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                # Add channel labels
                for i, ch in enumerate(common_channels):
                    plt.annotate(ch, (encoder_scores[i], predictor_scores[i]), 
                                fontsize=8, alpha=0.8)
                
                plt.xlabel('Encoder R² Score')
                plt.ylabel('Predictor R² Score')
                plt.title(f'{model.capitalize()} Model - Encoder vs Predictor Performance')
                plt.grid(True, alpha=0.3)
                
                # Set equal aspect ratio
                plt.axis('equal')
                plt.tight_layout()
                
                comp_path = self.output_dir / f"{model}_encoder_vs_predictor.png"
                plt.savefig(comp_path)
                
                if self.use_wandb:
                    self.wandb_run.log({f"{model}_encoder_vs_predictor": wandb.Image(plt)})
                    
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Probe PLDM models to predict state channels")
    
    # Config file and model paths
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration file")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained PLDM models")
    parser.add_argument("--output_dir", type=str, default="channel_probing_results",
                        help="Directory to save probing results")
    
    # Probing settings
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train probers")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training probers")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for training probers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model selection
    parser.add_argument("--probe_dynamics", action="store_true", default=True,
                        help="Probe dynamics model")
    parser.add_argument("--probe_reward", action="store_true", default=True,
                        help="Probe reward model")
    parser.add_argument("--probe_encoder", action="store_true", default=True,
                        help="Probe state encoder representations")
    parser.add_argument("--probe_predictor", action="store_true", default=False,
                        help="Probe dynamics predictor representations")
    
    # Visualization and logging
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Generate visualizations of probing results")
    parser.add_argument("--no_visualize", action="store_false", dest="visualize",
                        help="Skip generating visualizations")
    parser.add_argument("--save_weights", action="store_true", default=True,
                        help="Save prober model weights to disk")
    parser.add_argument("--no_save_weights", action="store_false", dest="save_weights",
                        help="Skip saving prober model weights (saves disk space)")
    parser.add_argument("--save_visualizations", action="store_true", default=True,
                        help="Save visualization images to disk")
    parser.add_argument("--no_save_visualizations", action="store_false", dest="save_visualizations",
                        help="Skip saving visualization images (saves disk space)")
    parser.add_argument("--max_vis_samples", type=int, default=5,
                        help="Maximum number of samples to visualize")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log results to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="pldm-channel-probing",
                        help="WandB project name")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logger(level=log_level)
    
    # Load configuration
    config = get_default_config()
    if args.config:
        try:
            loaded_config = load_config(args.config)
            config = merge_configs(config, loaded_config)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return
    
    # Create a probing config
    probing_config = ProbingConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        probe_dynamics=args.probe_dynamics,
        probe_reward=args.probe_reward,
        probe_encoder=args.probe_encoder,
        probe_predictor=args.probe_predictor,
        visualize=args.visualize,
        save_weights=args.save_weights,
        save_visualizations=args.save_visualizations,
        max_vis_samples=args.max_vis_samples,
        use_wandb=args.use_wandb
    )
    
    # Initialize WandB logging if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config={**config, **vars(args)}
            )
            logger.info(f"WandB initialized: {wandb_run.name} ({wandb_run.url})")
        except ImportError:
            logger.warning("Could not import wandb. WandB logging disabled.")
        except Exception as e:
            logger.error(f"Error initializing WandB: {e}")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load trained models
    try:
        # Initialize trainer with the saved models
        trainer = PLDMTrainer(
            data_path=config["data"]["train_data_path"],
            output_dir=args.model_dir,
            model_type=config["model"]["type"],
            batch_size=config["training"]["batch_size"],
            device=device,
            wandb_run=wandb_run,
            config=config
        )
        
        # Load saved models
        trainer.load_model('dynamics')
        trainer.load_model('reward')
        
        logger.info("PLDM models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        if wandb_run:
            wandb_run.finish(exit_code=1)
        return
    
    # Initialize probing evaluator
    prober_evaluator = ProbingEvaluator(
        model=trainer,
        output_dir=args.output_dir,
        config=probing_config,
        device=device,
        wandb_run=wandb_run
    )
    
    # Train and evaluate probers
    try:
        results = prober_evaluator.train_probers_for_all_channels()
        logger.info("Channel probing completed successfully")
    except Exception as e:
        logger.error(f"Error during probing: {e}")
        if wandb_run:
            wandb_run.finish(exit_code=1)
        return
    
    # Finish WandB run
    if wandb_run:
        wandb_run.finish()
        logger.info("WandB run finished")


if __name__ == "__main__":
    main() 