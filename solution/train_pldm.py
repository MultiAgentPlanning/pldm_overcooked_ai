import argparse
import os
import sys
from pathlib import Path
import logging # Import logging
import wandb   # Import wandb
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import torch.nn as nn
import gc  # Import garbage collection

os.environ['WANDB_IGNORE_GLOBS'] = '*.pem'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['WANDB_DISABLE_ARTIFACTS'] = 'true'

# Add the parent directory to the path so we can import the pldm module
sys.path.append(str(Path(__file__).parent.parent))

from solution.pldm.trainer import PLDMTrainer
from solution.pldm.config import (
    load_config, 
    save_config, 
    get_default_config, 
    create_default_config_file, 
    merge_configs
)
from solution.pldm.utils import setup_logger, set_seeds # Import set_seeds function
from solution.pldm.data_processor import OvercookedDataset
from solution.pldm import Planner
from solution.pldm.utils import parse_state
from solution.pldm.prober import Prober, GridProber, VectorProber
from solution.probe_pldm import ProbingEvaluator, ProbingConfig, StateChannels

# Get a logger for this module
logger = logging.getLogger(__name__)

def print_model_architecture(model, model_name="Model"):
    """
    Print detailed architecture information for a PyTorch model.
    
    Args:
        model: PyTorch model to analyze
        model_name: Name to display for this model
    """
    # Title with model name
    header = f"{'=' * 30} {model_name} ARCHITECTURE {'=' * 30}"
    print(f"\n{header}")
    
    # Print model summary
    print(model)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print detailed layer information if available
    print("\nDetailed layer shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    
    # Print footer
    print("=" * len(header) + "\n")

def evaluate_planner(config, model_dir, device, wandb_run=None, print_arch=False):
    """
    Evaluate the trained models using planning-based evaluation.

    Args:
        config: The configuration dictionary
        model_dir: Directory containing the trained models
        device: Device to use for evaluation
        wandb_run: Optional WandB run object for logging results
        print_arch: Whether to print architecture details
    
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Starting planner-based evaluation...")
    
    # Load parameters for evaluation
    data_path = config["data"]["train_data_path"]
    num_samples = config["testing"]["num_samples"]
    planning_horizon = config["testing"]["planning_horizon"]
    planning_samples = config["testing"]["planning_samples"]
    model_type = config["model"]["type"]
    
    # Special handling for seed when using CUDA devices
    seed = config.get("seed", 42)
    if device.type == 'cuda' and seed is not None:
        logger.warning("Using a seed with CUDA devices might cause issues with the random generator.")
        logger.warning("If you encounter 'Expected a 'cuda' device type for generator but found 'cpu'' errors,")
        logger.warning("the planner.py fix should handle this. If not, try with --seed None or updating PyTorch.")
    
    logger.info(f"Evaluating with planning horizon {planning_horizon}, {planning_samples} samples")

    # Load the saved models - using PLDMTrainer's saved models
    try:
        # Load models directly
        dynamics_model_path = Path(model_dir) / "dynamics_model.pt"
        reward_model_path = Path(model_dir) / "reward_model.pt"
        
        if not dynamics_model_path.exists() or not reward_model_path.exists():
            logger.error(f"Model files not found in {model_dir}. Skipping evaluation.")
            return {}
        
        # Initialize models using config parameters
        if model_type == 'grid':
            from solution.pldm.state_encoder import GridStateEncoder
            from solution.pldm.dynamics_predictor import GridDynamicsPredictor
            from solution.pldm.reward_predictor import GridRewardPredictor
            
            state_encoder = GridStateEncoder(
                grid_height=config["model"].get("grid_height"),
                grid_width=config["model"].get("grid_width")
            )
            
            num_channels = state_encoder.num_channels
            
            dynamics_model = GridDynamicsPredictor(
                state_embed_dim=config["model"]["state_embed_dim"],
                action_embed_dim=config["model"]["action_embed_dim"],
                num_actions=config["model"]["num_actions"],
                hidden_dim=config["model"]["dynamics_hidden_dim"],
                num_channels=num_channels,
                grid_height=config["model"].get("grid_height"),
                grid_width=config["model"].get("grid_width")
            ).to(device)
            
            reward_model = GridRewardPredictor(
                state_embed_dim=config["model"]["state_embed_dim"],
                action_embed_dim=config["model"]["action_embed_dim"],
                num_actions=config["model"]["num_actions"],
                hidden_dim=config["model"]["reward_hidden_dim"],
                num_channels=num_channels,
                grid_height=config["model"].get("grid_height"),
                grid_width=config["model"].get("grid_width")
            ).to(device)
        
        elif model_type == 'vector':
            from solution.pldm.state_encoder import VectorStateEncoder
            from solution.pldm.dynamics_predictor import VectorDynamicsPredictor
            from solution.pldm.reward_predictor import VectorRewardPredictor
            
            state_encoder = VectorStateEncoder(
                grid_height=config["model"].get("grid_height"),
                grid_width=config["model"].get("grid_width")
            )
            
            # For vector models, we'll need to determine state_dim later
            # We'll use a placeholder value of 0 that will be overridden
            dynamics_model = VectorDynamicsPredictor(
                state_dim=0,  # This will be overridden by the loaded state_dict
                action_embed_dim=config["model"]["action_embed_dim"],
                num_actions=config["model"]["num_actions"],
                hidden_dim=config["model"]["dynamics_hidden_dim"]
            ).to(device)
            
            reward_model = VectorRewardPredictor(
                state_dim=0,  # This will be overridden by the loaded state_dict
                action_embed_dim=config["model"]["action_embed_dim"],
                num_actions=config["model"]["num_actions"],
                hidden_dim=config["model"]["reward_hidden_dim"]
            ).to(device)
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return {}
        
        # Load model weights
        from solution.test_pldm import load_model_with_dynamic_layers
        
        load_model_with_dynamic_layers(dynamics_model, dynamics_model_path, device)
        load_model_with_dynamic_layers(reward_model, reward_model_path, device)
        
        # Set models to evaluation mode
        dynamics_model.eval()
        reward_model.eval()
        
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return {}
    
    # Initialize planner
    logger.info("Initializing planner...")
    num_actions = config["model"]["num_actions"]
    planner = Planner(
        dynamics_model=dynamics_model,
        reward_model=reward_model,
        state_encoder=state_encoder,
        num_actions=num_actions,
        planning_horizon=planning_horizon,
        num_samples=planning_samples,
        device=device,
        seed=seed
    )
    logger.info("Planner initialized.")
    
    # Print planner details if requested
    if print_arch:
        # Print planner configuration
        header = f"{'=' * 30} PLANNER CONFIGURATION {'=' * 30}"
        print(f"\n{header}")
        print(f"Planning Horizon: {planning_horizon}")
        print(f"Planning Samples: {planning_samples}")
        print(f"Number of Actions: {num_actions}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print(f"Model Type: {model_type}")
        print("=" * len(header) + "\n")
        
        # Print dynamics and reward models used in the planner
        print_model_architecture(dynamics_model, "PLANNER DYNAMICS MODEL")
        print_model_architecture(reward_model, "PLANNER REWARD MODEL")
    
    # Load the dataset for evaluation
    logger.info("Loading dataset for evaluation...")
    try:
        full_dataset = OvercookedDataset(
            data_path=data_path,
            state_encoder_type=model_type,
            max_samples=None  # Load everything for accurate episode info
        )
        logger.info(f"Dataset loaded with {len(full_dataset)} transitions.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return {}
    
    # Determine number of samples to evaluate
    num_eval_samples = min(num_samples, len(full_dataset))
    eval_indices = range(num_eval_samples)
    logger.info(f"Evaluating on {num_eval_samples} starting states...")
    
    # Prepare for evaluation
    planner_predicted_rewards = []
    actual_cumulative_rewards = []
    
    # Evaluate on each sample
    for i in eval_indices:
        logger.info(f"Processing sample {i+1}/{num_eval_samples}")
        
        try:
            # Load the current state from the CSV
            df = pd.read_csv(data_path, skiprows=range(1, i + 1), nrows=1)
            if len(df) == 0:
                logger.warning(f"Could not read sample {i} from data file.")
                continue
                
            current_state_json = df.iloc[0]['state']
            current_state_dict = parse_state(current_state_json)
            
            # Run the planner
            planned_action_indices, predicted_reward = planner.plan(current_state_dict)
            planner_predicted_rewards.append(predicted_reward)
            
            # Calculate the actual cumulative reward
            from solution.test_pldm import get_actual_cumulative_reward
            actual_reward = get_actual_cumulative_reward(full_dataset, i, planning_horizon)
            actual_cumulative_rewards.append(actual_reward)
            
            # Log detailed step info if debugging
            logger.debug(f"  Predicted reward: {predicted_reward:.4f}, Actual reward: {actual_reward:.4f}")
            
        except Exception as e:
            logger.error(f"Error during planning for sample {i}: {e}")
            continue
    
    # Calculate evaluation metrics
    evaluation_metrics = {}
    
    valid_predicted_rewards = [r for r in planner_predicted_rewards if not np.isnan(r)]
    valid_actual_rewards = [r for r in actual_cumulative_rewards if not np.isnan(r)]
    num_valid_samples = len(valid_predicted_rewards)
    
    if num_valid_samples > 0:
        avg_predicted_reward = np.mean(valid_predicted_rewards)
        avg_actual_reward = np.mean(valid_actual_rewards)
        reward_difference = np.mean(np.array(valid_predicted_rewards) - np.array(valid_actual_rewards)) if len(valid_actual_rewards) == num_valid_samples else np.nan
        
        evaluation_metrics = {
            'planner_avg_predicted_reward': avg_predicted_reward,
            'planner_avg_actual_reward': avg_actual_reward,
            'planner_reward_difference': reward_difference,
            'planner_valid_samples': num_valid_samples,
            'planner_total_samples': num_eval_samples
        }
        
        # Log to WandB if available
        if wandb_run is not None:
            wandb_run.log(evaluation_metrics)
            
            # Create a histogram of rewards if there are enough samples
            if len(valid_predicted_rewards) >= 10:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    plt.hist(valid_predicted_rewards, alpha=0.5, label='Predicted')
                    plt.hist(valid_actual_rewards, alpha=0.5, label='Actual')
                    plt.legend()
                    plt.title('Distribution of Rewards')
                    plt.xlabel('Reward')
                    plt.ylabel('Count')
                    wandb_run.log({"reward_distribution": wandb.Image(plt)})
                    plt.close()
                except Exception as e:
                    logger.error(f"Error creating reward histogram: {e}")
        
        # Log the results
        logger.info("\n--- Planner Evaluation Results ---")
        logger.info(f"Average Predicted Reward: {avg_predicted_reward:.4f}")
        logger.info(f"Average Actual Reward: {avg_actual_reward:.4f}")
        logger.info(f"Average Difference (Predicted - Actual): {reward_difference:.4f}")
        logger.info(f"Valid Samples: {num_valid_samples}/{num_eval_samples}")
    else:
        logger.warning("No valid planner evaluation samples.")
    
    return evaluation_metrics

def run_probing(config, model_dir, device, wandb_run=None, print_arch=False):
    """
    Run probing analysis on trained models to evaluate their state channel representations.
    
    Args:
        config: The configuration dictionary
        model_dir: Directory containing the trained models
        device: Device to use for probing
        wandb_run: Optional WandB run object for logging results
        print_arch: Whether to print architecture details
    
    Returns:
        Dictionary containing probing metrics
    """
    logger.info("Starting state channel representation probing analysis...")
    
    # Check if probing configuration exists
    if "probing" not in config:
        logger.warning("No probing configuration found in config. Using default probing settings.")
        config["probing"] = get_default_config()["probing"]
    
    # Extract probing configuration
    probing_config = config.get("probing", {})
    
    # Set random seed for reproducibility in probing
    seed = config.get("seed", 42)
    logger.info(f"Setting probing seed to {seed} for reproducibility")
    set_seeds(seed)  # This will set seeds for random, numpy, and torch
    
    # Create probing output directory 
    probing_dir = Path(model_dir) / "channel_probing"
    os.makedirs(probing_dir, exist_ok=True)
    
    # Get the models to probe based on config
    probe_dynamics = probing_config.get("probe_dynamics", True)
    probe_reward = probing_config.get("probe_reward", True)
    probe_encoder = probing_config.get("probe_encoder", True)
    probe_predictor = probing_config.get("probe_predictor", False)
    
    if not probe_dynamics and not probe_reward:
        logger.warning("Neither dynamics nor reward model selected for probing. Skipping.")
        return {}
    
    # Create ProbingConfig instance
    probe_config = ProbingConfig(
        lr=probing_config.get("lr", 1e-3),
        epochs=probing_config.get("epochs", 30),
        batch_size=probing_config.get("batch_size", 64),
        seed=seed,
        probe_dynamics=probe_dynamics,
        probe_reward=probe_reward,
        probe_encoder=probe_encoder,
        probe_predictor=probe_predictor,
        visualize=probing_config.get("visualize", True),
        save_weights=probing_config.get("save_weights", True),
        save_visualizations=probing_config.get("save_visualizations", True),
        max_vis_samples=probing_config.get("max_vis_samples", 5),
        use_wandb=wandb_run is not None
    )
    
    # Log probing configuration for reproducibility
    logger.info(f"Probing configuration: batch_size={probe_config.batch_size}, "
                f"epochs={probe_config.epochs}, seed={seed}")
    
    # Load the saved models using PLDMTrainer
    try:
        # Set a smaller batch size for probing to reduce memory usage
        probing_batch_size = min(probe_config.batch_size, 32)  # Use a smaller batch size if configured one is large
        logger.info(f"Using batch size {probing_batch_size} for probing to manage memory usage")
        
        # Initialize trainer with the saved models
        trainer = PLDMTrainer(
            data_path=config["data"]["train_data_path"],
            output_dir=model_dir,
            model_type=config["model"]["type"],
            batch_size=probing_batch_size,
            device=device,
            wandb_run=wandb_run,
            seed=seed,  # Explicitly pass seed to ensure data loaders are consistent
            config=config  # Pass the full config
        )
        
        # Load saved models
        trainer.load_model('dynamics')
        trainer.load_model('reward')
        
        logger.info("PLDM models loaded successfully for probing")
    except Exception as e:
        logger.error(f"Error loading models for probing: {e}")
        logger.exception("Probing exception details:")
        return {}
    
    # Initialize prober evaluator
    prober_evaluator = ProbingEvaluator(
        model=trainer,
        output_dir=str(probing_dir),
        config=probe_config,
        device=device,
        wandb_run=wandb_run
    )
    
    # Save probing configuration for reproducibility
    probing_config_path = probing_dir / "probe_config.json"
    with open(probing_config_path, 'w') as f:
        json.dump({
            'seed': seed,
            'batch_size': probe_config.batch_size,
            'epochs': probe_config.epochs,
            'lr': probe_config.lr,
            'probe_dynamics': probe_config.probe_dynamics,
            'probe_reward': probe_config.probe_reward,
            'probe_encoder': probe_config.probe_encoder,
            'probe_predictor': probe_config.probe_predictor,
            'visualize': probe_config.visualize,
            'model_type': config["model"]["type"]
        }, f, indent=2)
    logger.info(f"Saved probing configuration to {probing_config_path}")
    
    # Train and evaluate probers
    try:
        logger.info(f"Training and evaluating state channel probers with {probe_config.epochs} epochs...")
        
        # List the channels being probed
        channel_names = StateChannels.get_channel_names()
        logger.info(f"Probing {len(channel_names)} state channels: {', '.join(channel_names[:5])}...")
        
        # Train and evaluate all probers
        results = prober_evaluator.train_probers_for_all_channels()
        
        # Print prober architecture if requested
        if print_arch and results:
            # Get a sample prober
            sample_prober_key = list(prober_evaluator.probers.keys())[0]
            sample_prober = prober_evaluator.probers[sample_prober_key]
            print_model_architecture(sample_prober, f"CHANNEL PROBER ARCHITECTURE")
        
        # Extract key metrics from results for overall summary
        summary_metrics = {}
        for key, metrics in results.items():
            model_type, repr_type, channel_name = key
            summary_key = f"probe_{model_type}_{repr_type}_{channel_name}"
            
            # Include the main metrics
            summary_metrics[f"{summary_key}_mean_loss"] = metrics.get('mean_loss', 0)
            summary_metrics[f"{summary_key}_rmse"] = metrics.get('rmse', 0) 
            summary_metrics[f"{summary_key}_r2_score"] = metrics.get('r2_score', 0)
        
        # Clean up memory
        logger.info("Cleaning up memory after probing...")
        # Remove large objects
        prober_evaluator = None
        trainer = None
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        return summary_metrics
        
    except Exception as e:
        logger.error(f"Error during probing: {e}")
        logger.exception("Probing exception details:")
        # Clean up even on error
        trainer = None
        prober_evaluator = None
        gc.collect()
        torch.cuda.empty_cache()
        return {}

def main():
    parser = argparse.ArgumentParser(description="Train PLDM models for Overcooked-AI")
    
    # Config file parameters
    parser.add_argument("--config", type=str, 
                        help="Path to configuration file (YAML or JSON)")
    parser.add_argument("--create_config", type=str, 
                        help="Create a default configuration file at the specified path and exit")
    
    # Optional parameters to override config
    parser.add_argument("--data_path", type=str,
                        help="Path to CSV dataset (overrides config)")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save models and logs (overrides config)")
    parser.add_argument("--model_type", type=str, choices=["grid", "vector"],
                        help="Type of model architecture (overrides config)")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for training (overrides config)")
    parser.add_argument("--max_samples", type=int,
                        help="Maximum number of samples to load for testing (overrides config)")
    parser.add_argument("--device", type=str,
                        help="Device to use for training (overrides config)")
    parser.add_argument("--test", action="store_true",
                        help="Test with a small subset of data (shortcut for testing configuration)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--print_model_arch", action="store_true", default=True,
                        help="Print detailed model architectures")

    # Workflow control arguments (to override config)
    parser.add_argument("--run_training", action="store_true", dest="run_training",
                        help="Run the training phase (overrides config)")
    parser.add_argument("--no_training", action="store_false", dest="run_training",
                        help="Skip the training phase (overrides config)")
    parser.add_argument("--run_probing", action="store_true", dest="run_probing",
                        help="Run the probing phase (overrides config)")
    parser.add_argument("--no_probing", action="store_false", dest="run_probing",
                        help="Skip the probing phase (overrides config)")
    parser.add_argument("--run_planning", action="store_true", dest="run_planning",
                        help="Run the planning phase (overrides config)")
    parser.add_argument("--no_planning", action="store_false", dest="run_planning",
                        help="Skip the planning phase (overrides config)")
    
    # Set default values to None for workflow flags so we can detect if they're explicitly set
    parser.set_defaults(run_training=None, run_probing=None, run_planning=None)

    # Legacy flags (kept for backward compatibility)
    parser.add_argument("--skip_evaluation", action="store_true", 
                        help="[Deprecated] Skip post-training planner evaluation (use --no_planning instead)")
    parser.add_argument("--skip_probing", action="store_true",
                        help="[Deprecated] Skip post-training representation probing (use --no_probing instead)")
    
    # Logging and WandB arguments
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to save log output to a file.")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--probing_targets", type=str, default=None,
                        help="Comma-separated list of probing targets (e.g., 'state,reward,agent_pos')")
    parser.add_argument("--probing_epochs", type=int, default=None,
                        help="Number of epochs for probing (overrides config)")
    parser.add_argument("--save_prober_weights", action="store_true", dest="save_prober_weights", default=True,
                        help="Save prober model weights to disk")
    parser.add_argument("--no_save_prober_weights", action="store_false", dest="save_prober_weights",
                        help="Skip saving prober model weights (saves disk space)")
    parser.add_argument("--save_visualizations", action="store_true", dest="save_visualizations", default=True,
                        help="Save visualization images to disk")
    parser.add_argument("--no_save_visualizations", action="store_false", dest="save_visualizations",
                        help="Skip saving visualization images (saves disk space)")

    # New command-line arguments
    parser.add_argument("--dynamics_epochs", type=int, default=None,
                        help="Number of epochs to train dynamics model (overrides config)")
    parser.add_argument("--reward_epochs", type=int, default=None,
                        help="Number of epochs to train reward model (overrides config)")
    parser.add_argument("--train_dynamics", action="store_true", dest="train_dynamics", default=True,
                        help="Train the dynamics model")
    parser.add_argument("--no_train_dynamics", action="store_false", dest="train_dynamics",
                        help="Skip training the dynamics model")
    parser.add_argument("--train_reward", action="store_true", dest="train_reward", default=True,
                        help="Train the reward model")
    parser.add_argument("--no_train_reward", action="store_false", dest="train_reward",
                        help="Skip training the reward model")
    
    args = parser.parse_args()
    
    # --- Setup Logging --- 
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    setup_logger(level=log_level_map.get(args.log_level, logging.INFO), log_file=args.log_file)
    
    logger.info("Starting PLDM training script.")
    
    # Create default config file if requested
    if args.create_config:
        create_default_config_file(args.create_config)
        logger.info(f"Default configuration created at {args.create_config}. Exiting.")
        return
    
    # --- Load Configuration --- 
    logger.info("Loading configuration...")
    config = get_default_config()
    if args.config:
        try:
            loaded_config = load_config(args.config)
            config = merge_configs(config, loaded_config)
            logger.info(f"Loaded configuration from {args.config}")
        except FileNotFoundError:
            logger.error(f"Config file not found at {args.config}. Using default config.")
        except Exception as e:
            logger.error(f"Error loading config file {args.config}: {e}. Using default config.")
            
    # Override config with command-line arguments if they were provided
    override_config = {}
    if args.data_path:
        override_config.setdefault("data", {})["train_data_path"] = args.data_path
    if args.output_dir:
        override_config.setdefault("training", {})["output_dir"] = args.output_dir
    if args.model_type:
        override_config.setdefault("model", {})["type"] = args.model_type
    if args.batch_size is not None:
        override_config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.max_samples is not None:
        override_config.setdefault("data", {})["max_samples"] = args.max_samples
    if args.device:
        override_config.setdefault("training", {})["device"] = args.device
    
    # Override probing settings if provided
    if args.probing_targets:
        override_config.setdefault("probing", {})["targets"] = args.probing_targets.split(',')
    if args.probing_epochs is not None:
        override_config.setdefault("probing", {})["epochs"] = args.probing_epochs
    if args.save_prober_weights is not None:
        override_config.setdefault("probing", {})["save_weights"] = args.save_prober_weights
    if args.save_visualizations is not None:
        override_config.setdefault("probing", {})["save_visualizations"] = args.save_visualizations
    
    # Override training settings if provided
    if args.dynamics_epochs is not None:
        override_config.setdefault("training", {})["dynamics_epochs"] = args.dynamics_epochs
    if args.reward_epochs is not None:
        override_config.setdefault("training", {})["reward_epochs"] = args.reward_epochs
    if args.train_dynamics is not None:
        override_config.setdefault("training", {})["train_dynamics"] = args.train_dynamics
    if args.train_reward is not None:
        override_config.setdefault("training", {})["train_reward"] = args.train_reward
    
    # Override workflow settings if command-line flags were used
    # Handle workflow settings, prioritizing the new flags over legacy ones
    workflow_config = override_config.setdefault("workflow", {})
    
    # Handle new workflow flags (if explicitly set)
    if args.run_training is not None:
        workflow_config["run_training"] = args.run_training
    if args.run_probing is not None:
        workflow_config["run_probing"] = args.run_probing
    if args.run_planning is not None:
        workflow_config["run_planning"] = args.run_planning
    
    # Handle legacy flags (with lower priority than new flags)
    if args.skip_probing and args.run_probing is None:
        workflow_config["run_probing"] = False
    if args.skip_evaluation and args.run_planning is None:
        workflow_config["run_planning"] = False
    
    # Apply test settings if --test flag is set
    if args.test:
        logger.info("Applying --test settings (reduced epochs, samples, etc.)")
        test_overrides = {
            "data": {"max_samples": 1000 if args.max_samples is None else args.max_samples},
            "training": {"dynamics_epochs": 2, "reward_epochs": 2, "log_interval": 5}
        }
        # Merge test settings carefully, respecting other command-line args
        test_conf_temp = merge_configs(override_config, test_overrides)
        override_config = merge_configs(override_config, test_conf_temp) # Ensure overrides take precedence
        # Ensure max_samples from --test is used if --max_samples wasn't provided
        if args.max_samples is None:
             override_config.setdefault("data", {})["max_samples"] = 1000

    # Command-line overrides for seed and wandb usage (have priority over config)
    if args.seed is not None:
        override_config["seed"] = args.seed  # Seed at top level
    if args.use_wandb:
        override_config.setdefault("wandb", {})["use_wandb"] = True

    # Merge final overrides
    config = merge_configs(config, override_config)
    logger.info("Configuration loaded and merged.")
    # Log the final configuration
    logger.debug(f"Final configuration: {json.dumps(config, indent=2)}")
    
    # --- Create output directory and save the used configuration
    output_dir = Path(config["training"]["output_dir"])
    try:
        os.makedirs(output_dir, exist_ok=True)
        save_config(config, output_dir / "training_config.yaml")
        logger.info(f"Output directory created: {output_dir}")
        logger.info(f"Final configuration saved to {output_dir / 'training_config.yaml'}")
    except Exception as e:
        logger.error(f"Could not create output directory or save config: {e}")
        return # Stop if we can't save config

    # --- Set Seeds for Reproducibility (from config or command line) ---
    seed = args.seed if args.seed is not None else config.get("seed", 42)  # Command-line has priority
    logger.info(f"Setting random seed to {seed}")
    set_seeds(seed)

    # --- Initialize WandB (if enabled) --- 
    wandb_run = None
    # Ensure wandb section exists in config with defaults if necessary
    config.setdefault("wandb", {}) 
    config["wandb"].setdefault("project", "pldm-overcooked") # Default project
    config["wandb"].setdefault("name", None) # Default name (WandB generates one)
    config["wandb"].setdefault("entity", None) # Default entity

    # Check if wandb should be used (from config or command line flag)
    use_wandb = config["wandb"].get("use_wandb", False) or args.use_wandb

    if use_wandb:
        try:
            # Use values from config
            wandb_project = config["wandb"]["project"]
            wandb_name = config["wandb"].get("name") # .get() handles None gracefully
            wandb_entity = config["wandb"].get("entity")

            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity, 
                name=wandb_name,
                config=config,  # Log the final configuration
                reinit=True     # Allow reinitialization if running in a notebook
            )
            logger.info(f"WandB initialized. Project: {wandb_project}, Name: {wandb_run.name}, Entity: {wandb_entity} ({wandb_run.url})")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            wandb_run = None # Ensure it's None if init fails
    else:
        logger.info("WandB logging disabled.")

    # --- Determine Device --- 
    device_str = config["training"]["device"]
    if device_str == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Get workflow settings from config
    run_training = config["workflow"].get("run_training", True)
    should_run_probing = config["workflow"].get("run_probing", True)
    run_planning = config["workflow"].get("run_planning", True)
    
    # --- Workflow Summary ---
    logger.info("\n=== PLDM Workflow ===")
    logger.info(f"1. Training: {'ENABLED' if run_training else 'DISABLED'}")
    logger.info(f"2. Probing: {'ENABLED' if should_run_probing else 'DISABLED'}")
    logger.info(f"3. Planning: {'ENABLED' if run_planning else 'DISABLED'}")
    logger.info("====================\n")
    
    # Only initialize trainer if we're running at least one phase
    if not (run_training or should_run_probing or run_planning):
        logger.warning("All workflow phases are disabled. Nothing to do.")
        if wandb_run:
            wandb_run.finish()
        return
    
    # --- Initialize Trainer if needed ---
    trainer = None
    if run_training or should_run_probing:
        logger.info("Initializing PLDMTrainer...")
        try:
            trainer = PLDMTrainer(
                data_path=config["data"]["train_data_path"],
                output_dir=config["training"]["output_dir"],
                model_type=config["model"]["type"],
                batch_size=config["training"]["batch_size"],
                lr=config["training"]["learning_rate"],
                state_embed_dim=config["model"]["state_embed_dim"],
                action_embed_dim=config["model"]["action_embed_dim"],
                num_actions=config["model"]["num_actions"],
                dynamics_hidden_dim=config["model"]["dynamics_hidden_dim"],
                reward_hidden_dim=config["model"]["reward_hidden_dim"],
                grid_height=config["model"].get("grid_height"), # Use .get for optional params
                grid_width=config["model"].get("grid_width"),
                device=device,
                max_samples=config["data"].get("max_samples"),
                num_workers=config["training"]["num_workers"],
                val_ratio=config["data"].get("val_ratio", 0.1),
                test_ratio=config["data"].get("test_ratio", 0.1),
                seed=seed,  # Pass the seed for reproducible data splitting
                # Pass wandb_run object to trainer if needed, or check wandb.run directly
                wandb_run=wandb_run,
                disable_artifacts=True,  # Disable WandB artifacts to avoid storage errors
                config=config  # Pass the entire config to the trainer
            )
            logger.info("PLDMTrainer initialized.")
            
            # Print model architectures if requested
            if args.print_model_arch and trainer.dynamics_model is not None and trainer.reward_model is not None:
                print_model_architecture(trainer.dynamics_model, "DYNAMICS MODEL")
                print_model_architecture(trainer.reward_model, "REWARD MODEL")
                
        except ValueError as ve:
             logger.error(f"Trainer initialization failed: {ve}")
             if wandb_run: wandb.finish(exit_code=1)
             return
        except Exception as e:
            logger.exception("An unexpected error occurred during trainer initialization.")
            if wandb_run: wandb.finish(exit_code=1)
            return
    
    # --- Run Workflow --- 
    try:
        # --- 1. Training Phase ---
        if run_training:
            logger.info("\n=== PHASE 1: TRAINING ===")
            logger.info("Starting model training...")
            trainer.train_all()
            logger.info(f"Training complete. Models saved to {config['training']['output_dir']}")
            
            # Free up memory after training
            if hasattr(trainer, 'train_loader'):
                logger.info("Freeing up training data to reduce memory usage...")
                # Clear data loaders
                trainer.train_loader = None
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Memory cleanup completed")
                
        elif trainer and not run_training:
            # If trainer is initialized but training is skipped, we need to load the models
            try:
                logger.info("Loading existing models (training phase skipped)...")
                trainer.load_model('dynamics')
                trainer.load_model('reward')
                logger.info("Models loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                logger.warning("Proceeding without trained models may cause issues in subsequent phases.")
        
        # --- 2. Probing Phase ---
        if should_run_probing:
            logger.info("\n=== PHASE 2: PROBING ===")
            logger.info("Running representation probing...")
            
            # Free the main trainer object if possible before creating a new one for probing
            if trainer and run_training:
                logger.info("Releasing main trainer to free memory before probing...")
                # Only keep the models, clear everything else
                dynamics_model = trainer.dynamics_model
                reward_model = trainer.reward_model
                trainer = None
                gc.collect()
                torch.cuda.empty_cache()
                
            probing_metrics = run_probing(
                config=config,
                model_dir=config["training"]["output_dir"],
                device=device,
                wandb_run=wandb_run,
                print_arch=args.print_model_arch
            )
            
            # Log a summary of the probing metrics
            if probing_metrics:
                logger.info("Representation probing complete.")
                if wandb_run:
                    # Create a summary table for WandB
                    wandb_run.log({
                        "probing_summary": wandb.Table(
                            columns=["Metric", "Value"],
                            data=[[k, v] for k, v in probing_metrics.items()]
                        )
                    })
            else:
                logger.warning("Probing did not return any metrics.")
        else:
            logger.info("\n=== PHASE 2: PROBING (SKIPPED) ===")
        
        # --- 3. Planning Phase ---
        if run_planning:
            logger.info("\n=== PHASE 3: PLANNING ===")
            logger.info("Running planner evaluation...")
            eval_metrics = evaluate_planner(
                config=config,
                model_dir=config["training"]["output_dir"],
                device=device,
                wandb_run=wandb_run,
                print_arch=args.print_model_arch
            )
            
            # Log a summary of the evaluation metrics
            if eval_metrics:
                logger.info("Planner evaluation complete.")
                if wandb_run:
                    # Create a summary table for WandB
                    wandb_run.log({
                        "evaluation_summary": wandb.Table(
                            columns=["Metric", "Value"],
                            data=[[k, v] for k, v in eval_metrics.items()]
                        )
                    })
            else:
                logger.warning("Planner evaluation did not return any metrics.")
        else:
            logger.info("\n=== PHASE 3: PLANNING (SKIPPED) ===")
        
        logger.info("\n=== WORKFLOW COMPLETE ===")
            
    except Exception as e:
        logger.exception("An error occurred during workflow execution.")
        if wandb_run: wandb.finish(exit_code=1)
        return

    # --- Finish WandB Run --- 
    if wandb_run:
        wandb.finish()
        logger.info("WandB run finished.")

if __name__ == "__main__":
    main() 