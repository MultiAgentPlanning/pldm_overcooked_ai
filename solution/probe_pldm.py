import argparse
import os
import sys
import json
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from solution.pldm.trainer import PLDMTrainer
from solution.pldm.config import load_config, merge_configs, get_default_config
from solution.pldm.utils import setup_logger, set_seeds
from solution.pldm.prober import Prober, GridProber, VectorProber
from solution.pldm.data_processor import get_overcooked_dataloaders

# Get logger for this module
logger = logging.getLogger(__name__)

class ProbeTarget:
    """Enumeration of different targets that can be probed."""
    STATE = "state"  # Predict the full state
    AGENT_POS = "agent_pos"  # Predict agent position
    OBJECTS = "objects"  # Predict object positions
    REWARD = "reward"  # Predict reward


class ProberEvaluator:
    """
    Evaluates representations of the PLDM model by training probers
    to predict various targets from the latent space.
    """
    def __init__(
        self,
        pldm_trainer: PLDMTrainer,
        output_dir: str,
        config: Dict[str, Any],
        device: torch.device = None,
        wandb_run=None,
    ):
        """
        Initialize a ProberEvaluator.
        
        Args:
            pldm_trainer: Trained PLDMTrainer instance
            output_dir: Directory to save probers and results
            config: Configuration dictionary
            device: Device to use for training/evaluation
            wandb_run: Optional WandB run for logging
        """
        self.pldm_trainer = pldm_trainer
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.config = config
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Extracting relevant settings from config
        self.batch_size = config.get("batch_size", 32)
        self.lr = config.get("lr", 1e-3)
        self.epochs = config.get("epochs", 10)
        self.model_type = config.get("model_type", pldm_trainer.model_type)
        
        # Extract seed for reproducibility
        self.seed = config.get("seed", 42)
        self._set_seed()
        
        # WandB logging setup
        self.wandb_run = wandb_run
        self.use_wandb = wandb_run is not None
        
        # Dictionary to store probers
        self.probers = {}
        
        logger.info(f"ProberEvaluator initialized with {self.model_type} model type and seed {self.seed}")
    
    def _set_seed(self):
        """Set random seed for reproducibility."""
        from solution.pldm.utils import set_seeds
        set_seeds(self.seed)
        logger.debug(f"Seed set to {self.seed} in ProberEvaluator")
    
    def create_prober(self, target: str, input_dim: int, output_dim: int):
        """
        Create an appropriate prober based on model type and target.
        
        Args:
            target: Target to probe (from ProbeTarget)
            input_dim: Dimension of input representation
            output_dim: Dimension of output prediction
            
        Returns:
            Prober model instance
        """
        # We're dealing with vector embeddings (state_embed_dim) returned by models
        # Use the VectorProber regardless of the original model type
        logger.info(f"Creating prober with input_dim={input_dim}, output_dim={output_dim}")
        return VectorProber(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(128, 64)
        )
    
    def get_representation_and_target(self, batch, target: str, model_type: str):
        """
        Extract the representation and corresponding target from a batch.
        
        Args:
            batch: Batch of data
            target: Target to probe
            model_type: 'dynamics' or 'reward' to specify which model's representations to use
            
        Returns:
            Tuple of (representation, target)
        """
        state, action, next_state, reward = batch
        
        # Move to device
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        if isinstance(reward, torch.Tensor):
            reward = reward.to(self.device)
        
        # Get model
        if model_type == 'dynamics':
            model = self.pldm_trainer.dynamics_model
        else:  # reward
            model = self.pldm_trainer.reward_model
        
        # Get representation (latent encoding)
        with torch.no_grad():
            try:
                # Get the embedding from the model's encoder if available
                if hasattr(model, 'encode'):
                    representation = model.encode(state, action)
                elif hasattr(model, 'encoder') and callable(getattr(model, 'encoder')):
                    representation = model.encoder(state)
                else:
                    # Use the model's forward pass with return_embedding=True
                    representation = model(state, action, return_embedding=True)
                
                logger.debug(f"Got representation of shape {representation.shape}")
            except Exception as e:
                logger.error(f"Error getting representation: {e}")
                raise
        
        # Prepare target based on requested probe target
        try:
            if target == ProbeTarget.STATE:
                # For state prediction, we need to handle different state formats
                if self.model_type == "grid" and next_state.dim() == 4:
                    # For grid states, flatten to make it easier to predict
                    batch_size = next_state.size(0)
                    target_tensor = next_state.view(batch_size, -1)
                else:
                    # For vector states, use as is
                    target_tensor = next_state
                    
                logger.debug(f"State target shape: {target_tensor.shape}")
                
            elif target == ProbeTarget.REWARD:
                # Ensure reward is a tensor with shape [batch_size, 1]
                if isinstance(reward, torch.Tensor):
                    if reward.dim() == 1:
                        target_tensor = reward.unsqueeze(1)
                    else:
                        target_tensor = reward
                else:
                    # Convert to tensor if not already
                    reward_np = np.array(reward).reshape(-1, 1)
                    target_tensor = torch.tensor(reward_np, dtype=torch.float32, device=self.device)
                
                logger.debug(f"Reward target shape: {target_tensor.shape}")
                
            elif target == ProbeTarget.AGENT_POS:
                # Extract agent positions
                target_tensor = self._extract_agent_pos(state)
                logger.debug(f"Agent position target shape: {target_tensor.shape}")
                
            else:
                raise ValueError(f"Unsupported probe target: {target}")
                
        except Exception as e:
            logger.error(f"Error preparing target {target}: {e}")
            raise
            
        return representation, target_tensor
    
    def _extract_agent_pos(self, state):
        """
        Extract agent position from state.
        This is a simplified placeholder that needs to handle both grid and vector states.
        
        Args:
            state: Batch of states
            
        Returns:
            Tensor of agent positions
        """
        # For simplicity in probing, we'll just return a placeholder position
        # This avoids indexing errors while still allowing the probing to run
        bs = state.size(0)
        pos = torch.zeros((bs, 2), device=self.device)
        
        if self.model_type == "grid" and state.dim() == 4:
            # Only attempt to extract real positions if we have grid states with expected dimensions
            try:
                # Assuming channel 0 and 1 represent agent positions
                agent1_channel = 0
                
                for i in range(bs):
                    # Find agent 1 position (first agent)
                    agent_mask = state[i, agent1_channel]
                    if agent_mask.sum() > 0:  # If agent is present
                        agent_pos = agent_mask.nonzero()
                        if agent_pos.size(0) > 0:  # If we found any positions
                            # Take the first occurrence
                            pos[i, 0] = agent_pos[0, 0]  # Y position (height)
                            pos[i, 1] = agent_pos[0, 1]  # X position (width)
            except Exception as e:
                logger.warning(f"Error extracting agent positions: {e}. Using zeros.")
                
        return pos
    
    def train_prober(
        self,
        target: str,
        model_type: str = 'dynamics',
        output_dim: int = None,
        train_loader=None,
        val_loader=None
    ):
        """
        Train a prober to predict a specific target from model representations.
        
        Args:
            target: Target to predict (from ProbeTarget)
            model_type: 'dynamics' or 'reward' to specify which model's representations to use
            output_dim: Dimension of the output (if None, inferred from data)
            train_loader: DataLoader for training (if None, use trainer's)
            val_loader: DataLoader for validation (if None, use trainer's)
            
        Returns:
            Trained prober
        """
        # Set seed for reproducibility
        self._set_seed()
        logger.info(f"Training {model_type} prober for target: {target} with seed {self.seed}")
        
        # Get data loaders
        if train_loader is None or val_loader is None:
            if not hasattr(self.pldm_trainer, 'train_loader') or not hasattr(self.pldm_trainer, 'val_loader'):
                # If trainer doesn't have loaders, try to create them
                logger.info("Creating data loaders for prober training...")
                train_loader, val_loader, _ = get_overcooked_dataloaders(
                    data_path=self.config.get("data_path", ""),
                    state_encoder_type=self.model_type,
                    batch_size=self.batch_size,
                    val_ratio=0.1,
                    test_ratio=0.1,
                    max_samples=self.config.get("max_samples", None),
                    num_workers=self.config.get("num_workers", 4),
                    seed=self.seed  # Use consistent seed for reproducible data splits
                )
            else:
                train_loader = self.pldm_trainer.train_loader
                val_loader = self.pldm_trainer.val_loader
        
        # Get a batch to determine input and output dimensions
        batch = next(iter(train_loader))
        representation, target_tensor = self.get_representation_and_target(batch, target, model_type)
        
        input_dim = representation.size(1) if representation.dim() == 2 else representation.shape[1:]
        if output_dim is None:
            if target_tensor.dim() > 1:
                output_dim = target_tensor.size(1)
            else:
                output_dim = 1
        
        # Create prober with seed set for reproducible initialization
        self._set_seed()
        prober = self.create_prober(target, input_dim, output_dim)
        prober = prober.to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(prober.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Training
            prober.train()
            train_loss = 0
            train_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} Train"):
                representation, target_tensor = self.get_representation_and_target(batch, target, model_type)
                
                optimizer.zero_grad()
                
                prediction = prober(representation)
                loss = criterion(prediction, target_tensor)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            
            # Validation
            prober.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} Validate"):
                    representation, target_tensor = self.get_representation_and_target(batch, target, model_type)
                    
                    prediction = prober(representation)
                    loss = criterion(prediction, target_tensor)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Log to WandB
            if self.use_wandb:
                self.wandb_run.log({
                    f"prober_{model_type}_{target}/train_loss": avg_train_loss,
                    f"prober_{model_type}_{target}/val_loss": avg_val_loss,
                    f"prober_{model_type}_{target}/epoch": epoch + 1
                })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                prober_path = self.output_dir / f"prober_{model_type}_{target}.pt"
                
                torch.save({
                    'state_dict': prober.state_dict(),
                    'config': {
                        'input_dim': input_dim,
                        'output_dim': output_dim,
                        'model_type': self.model_type,
                        'target': target
                    }
                }, prober_path)
                
                logger.info(f"New best {model_type} prober for {target} saved to {prober_path}")
        
        # Store the prober
        self.probers[f"{model_type}_{target}"] = prober
        
        return prober
    
    def evaluate_prober(
        self,
        prober,
        target: str,
        model_type: str = 'dynamics',
        test_loader=None,
        visualize: bool = True
    ):
        """
        Evaluate a trained prober and visualize results.
        
        Args:
            prober: Trained prober model
            target: Target that was probed
            model_type: 'dynamics' or 'reward'
            test_loader: DataLoader for testing (if None, use trainer's test_loader)
            visualize: Whether to generate visualizations
        
        Returns:
            Dict of evaluation metrics
        """
        # Set seed for reproducible evaluation
        self._set_seed()
        logger.info(f"Evaluating {model_type} prober for target: {target} with seed {self.seed}")
        
        if test_loader is None:
            if not hasattr(self.pldm_trainer, 'test_loader'):
                logger.error("No test loader available for evaluation")
                return {}
            test_loader = self.pldm_trainer.test_loader
        
        prober.eval()
        criterion = torch.nn.MSELoss(reduction='none')
        
        all_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating prober"):
                representation, target_tensor = self.get_representation_and_target(batch, target, model_type)
                
                prediction = prober(representation)
                
                # Compute loss per sample/dimension
                loss = criterion(prediction, target_tensor)
                
                # Store results
                all_losses.append(loss)
                all_predictions.append(prediction.cpu())
                all_targets.append(target_tensor.cpu())
        
        # Concatenate results
        all_losses = torch.cat(all_losses, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        mean_loss = all_losses.mean().item()
        rmse = torch.sqrt(all_losses.mean()).item()
        
        # Calculate R^2 score
        target_var = torch.var(all_targets, dim=0, unbiased=False)
        unexplained_var = torch.var(all_targets - all_predictions, dim=0, unbiased=False)
        r2_score = 1 - (unexplained_var / target_var)
        mean_r2 = r2_score.mean().item()
        
        # Compile metrics
        metrics = {
            'mean_loss': mean_loss,
            'rmse': rmse,
            'r2_score': mean_r2
        }
        
        logger.info(f"Prober evaluation results for {model_type}_{target}:")
        logger.info(f"  Mean Loss: {mean_loss:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  R² Score: {mean_r2:.6f}")
        
        # Log to WandB
        if self.use_wandb:
            self.wandb_run.log({
                f"prober_{model_type}_{target}/test_loss": mean_loss,
                f"prober_{model_type}_{target}/test_rmse": rmse,
                f"prober_{model_type}_{target}/test_r2": mean_r2
            })
        
        # Visualize results if requested
        if visualize:
            self.visualize_prober_results(
                predictions=all_predictions,
                targets=all_targets,
                target_type=target,
                model_type=model_type
            )
        
        return metrics
    
    def visualize_prober_results(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_type: str,
        model_type: str
    ):
        """
        Generate visualizations of prober predictions vs. targets.
        
        Args:
            predictions: Tensor of predictions
            targets: Tensor of ground truth targets
            target_type: Type of target being predicted
            model_type: 'dynamics' or 'reward'
        """
        # Determine an appropriate number of samples to visualize
        n_samples = min(10, predictions.size(0))
        
        # Convert to numpy for plotting
        predictions_np = predictions[:n_samples].numpy()
        targets_np = targets[:n_samples].numpy()
        
        if target_type == ProbeTarget.REWARD:
            # For reward predictions, create a scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(targets_np, predictions_np, alpha=0.5)
            
            # Add diagonal line for perfect predictions
            min_val = min(targets_np.min(), predictions_np.min())
            max_val = max(targets_np.max(), predictions_np.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('True Reward')
            plt.ylabel('Predicted Reward')
            plt.title(f'{model_type.capitalize()} Model - Reward Predictions')
            
            # Save figure
            plt_path = self.output_dir / f"{model_type}_{target_type}_scatter.png"
            plt.savefig(plt_path)
            
            # Log to WandB
            if self.use_wandb:
                self.wandb_run.log({f"{model_type}_{target_type}_scatter": wandb.Image(plt)})
            
            plt.close()
            
        elif target_type in [ProbeTarget.AGENT_POS, ProbeTarget.OBJECTS]:
            # For position predictions, create plots showing predicted vs. actual positions
            for i in range(n_samples):
                plt.figure(figsize=(8, 8))
                
                # Plot actual positions
                plt.scatter(
                    targets_np[i, 0::2],  # x coordinates
                    targets_np[i, 1::2],  # y coordinates
                    color='blue',
                    label='True Position'
                )
                
                # Plot predicted positions
                plt.scatter(
                    predictions_np[i, 0::2],  # x coordinates
                    predictions_np[i, 1::2],  # y coordinates
                    color='red',
                    label='Predicted Position'
                )
                
                # Connect corresponding points with lines
                for j in range(0, targets_np.shape[1], 2):
                    plt.plot(
                        [targets_np[i, j], predictions_np[i, j]],
                        [targets_np[i, j+1], predictions_np[i, j+1]],
                        'k--', alpha=0.3
                    )
                
                plt.legend()
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.title(f'{model_type.capitalize()} Model - Position Predictions (Sample {i+1})')
                
                # Save figure
                plt_path = self.output_dir / f"{model_type}_{target_type}_sample{i+1}.png"
                plt.savefig(plt_path)
                
                # Log to WandB
                if self.use_wandb:
                    self.wandb_run.log({f"{model_type}_{target_type}_sample{i+1}": wandb.Image(plt)})
                
                plt.close()
        
        # For other target types, create appropriate visualizations
        else:
            logger.info(f"Visualization not implemented for target type: {target_type}")
    
    def train_all_probers(self):
        """
        Train probers for all combinations of models and targets.
        
        Returns:
            Dict mapping (model_type, target) to evaluation metrics
        """
        # Set seed for reproducible results
        self._set_seed()
        logger.info(f"Training all probers with seed {self.seed}")
        
        results = {}
        
        # Define models and targets to probe
        models = ['dynamics', 'reward']
        targets = [ProbeTarget.STATE, ProbeTarget.REWARD]
        
        # Add agent position if configured
        if self.config.get("probe_agent_pos", True):
            targets.append(ProbeTarget.AGENT_POS)
        
        # Save probing configuration for reproducibility
        probe_config_path = self.output_dir / "probe_config.json"
        with open(probe_config_path, 'w') as f:
            json.dump({
                'seed': self.seed,
                'model_type': self.model_type,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'lr': self.lr,
                'targets': [t for t in targets],
                'models': models
            }, f, indent=2)
        logger.info(f"Saved probing configuration to {probe_config_path}")
        
        # Train and evaluate probers for each combination
        for model_type in models:
            for target in targets:
                # Skip reward model probing reward (redundant)
                if model_type == 'reward' and target == ProbeTarget.REWARD:
                    continue
                
                # Set seed before each probing task
                self._set_seed()
                
                prober = self.train_prober(target, model_type)
                metrics = self.evaluate_prober(prober, target, model_type)
                
                # Include seed in metrics for reference
                metrics['seed'] = self.seed
                results[(model_type, target)] = metrics
        
        # Aggregate results
        self.aggregate_results(results)
        
        return results
    
    def aggregate_results(self, results: Dict):
        """
        Aggregate and summarize probing results.
        
        Args:
            results: Dict mapping (model_type, target) to metrics
        """
        # Create a summary table
        summary = []
        
        for (model_type, target), metrics in results.items():
            summary.append({
                'Model': model_type,
                'Target': target,
                'MSE': metrics['mean_loss'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2_score']
            })
        
        # Sort by model type then target
        summary.sort(key=lambda x: (x['Model'], x['Target']))
        
        # Print summary table
        logger.info("\n--- Probing Results Summary ---")
        logger.info(f"{'Model':<10} {'Target':<10} {'MSE':<10} {'RMSE':<10} {'R²':<10}")
        logger.info("-" * 50)
        
        for row in summary:
            logger.info(
                f"{row['Model']:<10} {row['Target']:<10} "
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
                columns=["Model", "Target", "MSE", "RMSE", "R²"]
            )
            
            for row in summary:
                wandb_table.add_data(
                    row['Model'],
                    row['Target'],
                    row['MSE'],
                    row['RMSE'],
                    row['R²']
                )
            
            self.wandb_run.log({"probing_summary": wandb_table})


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate probers for PLDM models")
    
    # Config file and model paths
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration file")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained PLDM models")
    parser.add_argument("--output_dir", type=str, default="probing_results",
                        help="Directory to save probing results")
    
    # Probing settings
    parser.add_argument("--targets", type=str, default="state,reward",
                        help="Comma-separated list of targets to probe (state,reward,agent_pos)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train probers")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training probers")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for training probers")
    
    # Visualization and logging settings
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of probing results")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log results to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="pldm-probing",
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
    
    # Add probing-specific settings to config
    probing_config = {
        "probing": {
            "targets": args.targets.split(","),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "visualize": args.visualize
        }
    }
    config = merge_configs(config, probing_config)
    
    # Initialize WandB logging if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=config
            )
            logger.info(f"WandB initialized: {wandb_run.name} ({wandb_run.url})")
        except ImportError:
            logger.warning("Could not import wandb. WandB logging disabled.")
        except Exception as e:
            logger.error(f"Error initializing WandB: {e}")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load trained model
    try:
        # Initialize trainer with the saved models
        trainer = PLDMTrainer(
            data_path=config["data"]["train_data_path"],
            output_dir=args.model_dir,
            model_type=config["model"]["type"],
            batch_size=config["training"]["batch_size"],
            device=device,
            wandb_run=wandb_run
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
    
    # Initialize prober evaluator
    prober_evaluator = ProberEvaluator(
        pldm_trainer=trainer,
        output_dir=args.output_dir,
        config=config["probing"],
        device=device,
        wandb_run=wandb_run
    )
    
    # Train and evaluate probers
    try:
        results = prober_evaluator.train_all_probers()
        logger.info("Probing completed successfully")
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