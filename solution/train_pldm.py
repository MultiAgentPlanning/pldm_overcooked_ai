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

# Get a logger for this module
logger = logging.getLogger(__name__)

def evaluate_planner(config, model_dir, device, wandb_run=None):
    """
    Evaluate the trained models using planning-based evaluation.

    Args:
        config: The configuration dictionary
        model_dir: Directory containing the trained models
        device: Device to use for evaluation
        wandb_run: Optional WandB run object for logging results
    
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
    seed = config.get("seed", 42)
    
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

    # Logging and WandB arguments
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to save log output to a file.")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip post-training planner evaluation.")
    
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
    
    # Create output directory and save the used configuration
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

    # --- Initialize Trainer --- 
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
            disable_artifacts=True  # Disable WandB artifacts to avoid storage errors
        )
        logger.info("PLDMTrainer initialized.")
    except ValueError as ve:
         logger.error(f"Trainer initialization failed: {ve}")
         if wandb_run: wandb.finish(exit_code=1)
         return
    except Exception as e:
        logger.exception("An unexpected error occurred during trainer initialization.")
        if wandb_run: wandb.finish(exit_code=1)
        return
    
    # --- Train Models --- 
    logger.info("Starting model training...")
    try:
        trainer.train_all(
            dynamics_epochs=config["training"]["dynamics_epochs"],
            reward_epochs=config["training"]["reward_epochs"],
            log_interval=config["training"]["log_interval"]
        )
        logger.info(f"Training complete. Models saved to {config['training']['output_dir']}")
        
        # Run planner-based evaluation after training (unless skipped)
        if not args.skip_evaluation:
            logger.info("Running post-training planner evaluation...")
            eval_metrics = evaluate_planner(
                config=config,
                model_dir=config["training"]["output_dir"],
                device=device,
                wandb_run=wandb_run
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
            logger.info("Post-training evaluation skipped.")
            
    except Exception as e:
        logger.exception("An error occurred during training or evaluation.")
        if wandb_run: wandb.finish(exit_code=1)
        return

    # --- Finish WandB Run --- 
    if wandb_run:
        wandb.finish()
        logger.info("WandB run finished.")

if __name__ == "__main__":
    main() 