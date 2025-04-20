import argparse
import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Add the parent directory to the path so we can import the pldm module
sys.path.append(str(Path(__file__).parent.parent))

from solution.pldm.utils import parse_state, parse_joint_action, get_action_index, set_seeds
from solution.pldm.state_encoder import GridStateEncoder, VectorStateEncoder
from solution.pldm.dynamics_predictor import GridDynamicsPredictor, VectorDynamicsPredictor
from solution.pldm.reward_predictor import GridRewardPredictor, VectorRewardPredictor
from solution.pldm.data_processor import OvercookedDataset
from solution.pldm.config import (
    load_config, 
    save_config, 
    get_default_config, 
    merge_configs
)
from solution.pldm import Planner


def load_hyperparameters(model_dir):
    """Load hyperparameters from the training config file or fallback to hyperparameters.json."""
    config_path = os.path.join(model_dir, "training_config.yaml")
    if os.path.exists(config_path):
        # Load from the saved training config
        config = load_config(config_path)
        return {
            'model_type': config["model"]["type"],
            'state_embed_dim': config["model"]["state_embed_dim"],
            'action_embed_dim': config["model"]["action_embed_dim"],
            'num_actions': config["model"]["num_actions"],
            'dynamics_hidden_dim': config["model"]["dynamics_hidden_dim"],
            'reward_hidden_dim': config["model"]["reward_hidden_dim"],
            'grid_height': config["model"]["grid_height"],
            'grid_width': config["model"]["grid_width"],
            'device': config["training"]["device"]
        }
    else:
        # Fallback to the old hyperparameters.json
        hyperparams_path = os.path.join(model_dir, 'hyperparameters.json')
        with open(hyperparams_path, 'r') as f:
            return json.load(f)


def load_models(model_dir):
    """Load the trained dynamics and reward models."""
    # Load hyperparameters
    hparams = load_hyperparameters(model_dir)
    
    # Initialize models based on model type
    if hparams['model_type'] == 'grid':
        state_encoder = GridStateEncoder()
        num_channels = state_encoder.num_channels
        
        dynamics_model = GridDynamicsPredictor(
            state_embed_dim=hparams['state_embed_dim'],
            action_embed_dim=hparams['action_embed_dim'],
            num_actions=hparams['num_actions'],
            hidden_dim=hparams['dynamics_hidden_dim'],
            num_channels=num_channels,
            grid_height=hparams['grid_height'],
            grid_width=hparams['grid_width']
        )
        
        reward_model = GridRewardPredictor(
            state_embed_dim=hparams['state_embed_dim'],
            action_embed_dim=hparams['action_embed_dim'],
            num_actions=hparams['num_actions'],
            hidden_dim=hparams['reward_hidden_dim'],
            num_channels=num_channels,
            grid_height=hparams['grid_height'],
            grid_width=hparams['grid_width']
        )
    
    elif hparams['model_type'] == 'vector':
        state_encoder = VectorStateEncoder()
        # We'd need a sample state to determine the vector dimension, but for testing
        # purposes we'll load from the trained model's input layer size
        dynamics_model = VectorDynamicsPredictor(
            state_dim=0,  # This will be overridden by the loaded state_dict
            action_embed_dim=hparams['action_embed_dim'],
            num_actions=hparams['num_actions'],
            hidden_dim=hparams['dynamics_hidden_dim']
        )
        
        reward_model = VectorRewardPredictor(
            state_dim=0,  # This will be overridden by the loaded state_dict
            action_embed_dim=hparams['action_embed_dim'],
            num_actions=hparams['num_actions'],
            hidden_dim=hparams['reward_hidden_dim']
        )
    
    else:
        raise ValueError(f"Unknown model type: {hparams['model_type']}")
    
    # Load model weights
    device = torch.device('cpu') if hparams.get('device', 'auto') == 'auto' else torch.device(hparams.get('device', 'cpu'))
    
    # Custom loading for the dynamics model
    load_model_with_dynamic_layers(dynamics_model, os.path.join(model_dir, 'dynamics_model.pt'), device)
    
    # Custom loading for the reward model
    load_model_with_dynamic_layers(reward_model, os.path.join(model_dir, 'reward_model.pt'), device)
    
    # Set models to evaluation mode
    dynamics_model.eval()
    reward_model.eval()
    
    return state_encoder, dynamics_model, reward_model, hparams


def load_model_with_dynamic_layers(model, model_path, device):
    """
    Custom model loading that handles dynamic layers in the model.
    
    Args:
        model: The model to load weights into
        model_path: Path to the saved model weights
        device: Device to load the model onto
    """
    print(f"Loading model from {model_path}")
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Initialize missing dynamic layers
    initialize_missing_layers(model, state_dict, device)
    
    # Filter out layers from the state dict that don't match current model
    filtered_state_dict = {}
    for name, param in state_dict.items():
        if name in model.state_dict():
            # Check if shapes match
            if param.shape == model.state_dict()[name].shape:
                filtered_state_dict[name] = param
            else:
                print(f"Skipping parameter {name} due to shape mismatch: " 
                      f"saved {param.shape} vs current {model.state_dict()[name].shape}")
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"Model loaded successfully with {len(filtered_state_dict)}/{len(state_dict)} parameters")


def initialize_missing_layers(model, state_dict, device):
    """
    Initialize missing dynamic layers in the model based on the state dict.
    
    Args:
        model: The model to initialize layers for
        state_dict: The saved state dict
        device: Device to create the layers on
    """
    # For GridDynamicsPredictor
    if isinstance(model, GridDynamicsPredictor):
        # Initialize encoder projection if needed
        if "encoder_proj.weight" in state_dict and model.encoder_proj is None:
            in_features = state_dict["encoder_proj.weight"].shape[1]
            out_features = state_dict["encoder_proj.weight"].shape[0]
            model.encoder_proj = torch.nn.Linear(in_features, out_features).to(device)
            print(f"Initialized encoder projection: {in_features} -> {out_features}")
        
        # Initialize output projection if needed
        if "output_proj.weight" in state_dict and model.output_proj is None:
            in_features = state_dict["output_proj.weight"].shape[1]
            out_features = state_dict["output_proj.weight"].shape[0]
            model.output_proj = torch.nn.Linear(in_features, out_features).to(device)
            print(f"Initialized output projection: {in_features} -> {out_features}")
            
        # Check if fc1 needs to be resized
        if "fc1.weight" in state_dict:
            saved_shape = state_dict["fc1.weight"].shape
            current_shape = model.fc1.weight.shape
            
            if saved_shape != current_shape:
                # Reinitialize fc1 with correct shape
                model.fc1 = torch.nn.Linear(saved_shape[1], saved_shape[0]).to(device)
                print(f"Resized fc1: {saved_shape[1]} -> {saved_shape[0]}")
    
    # For GridRewardPredictor
    elif isinstance(model, GridRewardPredictor):
        # Initialize encoder projection if needed
        if "encoder_proj.weight" in state_dict and model.encoder_proj is None:
            in_features = state_dict["encoder_proj.weight"].shape[1]
            out_features = state_dict["encoder_proj.weight"].shape[0]
            model.encoder_proj = torch.nn.Linear(in_features, out_features).to(device)
            print(f"Initialized encoder projection: {in_features} -> {out_features}")
            
        # Check if fc1 needs to be resized
        if "fc1.weight" in state_dict:
            saved_shape = state_dict["fc1.weight"].shape
            current_shape = model.fc1.weight.shape
            
            if saved_shape != current_shape:
                # Reinitialize fc1 with correct shape
                model.fc1 = torch.nn.Linear(saved_shape[1], saved_shape[0]).to(device)
                print(f"Resized fc1: {saved_shape[1]} -> {saved_shape[0]}")


def get_actual_cumulative_reward(dataset: OvercookedDataset, start_idx: int, horizon: int) -> float:
    """
    Calculates the actual cumulative reward from the dataset for a given horizon,
    respecting episode boundaries.

    Args:
        dataset: The OvercookedDataset instance containing transitions and episode info.
        start_idx: The starting index in the dataset's flattened transitions list.
        horizon: The number of steps (transitions) to sum rewards over.

    Returns:
        The actual cumulative reward (float), or 0.0 if calculation is not possible.
    """
    if not hasattr(dataset, 'episode_info') or not hasattr(dataset, 'transitions'):
        print("Warning: Dataset does not have expected episode_info or transitions. Cannot calculate actual reward.")
        return 0.0

    # Find the episode this start_idx belongs to
    episode_data = dataset.get_episode_info(start_idx)
    if episode_data is None:
        print(f"Warning: Could not find episode info for start_idx {start_idx}.")
        return 0.0

    episode_start, episode_end, _ = episode_data

    # Determine how many steps are actually possible within this episode from start_idx
    steps_left_in_episode = (episode_end - start_idx) + 1 
    
    # Determine the effective horizon
    effective_horizon = min(horizon, steps_left_in_episode)
    
    if effective_horizon <= 0:
        return 0.0

    cumulative_reward = 0.0
    try:
        # Sum rewards for the effective horizon
        for i in range(effective_horizon):
            # Access the reward from the stored transition tuple (index 3)
            reward_at_step = dataset.transitions[start_idx + i][3]
            cumulative_reward += reward_at_step
    except IndexError:
        print(f"Warning: Index out of bounds while calculating actual reward. start_idx={start_idx}, effective_horizon={effective_horizon}, dataset_len={len(dataset.transitions)}")
        return 0.0 # Return 0 if indices go out of bounds
    except Exception as e:
        print(f"Warning: Error calculating actual reward: {e}")
        return 0.0

    return cumulative_reward


def evaluate_models(data_path, model_dir, num_samples=100, planning_horizon=10, planning_samples=50, device=None, seed=None):
    """
    Evaluate the dynamics, reward, and planning models on a subset of the data.
    
    Args:
        data_path: Path to the test data
        model_dir: Directory containing the trained models
        num_samples: Number of samples (start states) to evaluate from the dataset
        planning_horizon: Horizon H for the planner's simulation.
        planning_samples: Number of trajectories the planner samples.
        device: Device to use for evaluation
        seed: Random seed for reproducibility
    """
    # Set seeds for reproducibility if provided
    if seed is not None:
        print(f"Setting random seed to {seed}")
        set_seeds(seed)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Load Models --- 
    print("Loading models...")
    try:
        state_encoder, dynamics_model, reward_model, hparams = load_models(model_dir)
        dynamics_model.to(device)
        reward_model.to(device)
        print("Models loaded.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # --- Initialize Planner --- 
    print("Initializing planner...")
    num_actions = hparams.get('num_actions', 6) # Default to 6 if not in hparams
    planner = Planner(
        dynamics_model=dynamics_model,
        reward_model=reward_model,
        state_encoder=state_encoder,
        num_actions=num_actions,
        planning_horizon=planning_horizon,
        num_samples=planning_samples,
        device=device,
        seed=seed  # Pass the seed to the planner
    )
    print("Planner initialized.")

    # --- Load Full Dataset for Accurate Episode Info ---
    print("Loading full dataset for episode boundary calculation...")
    try:
        full_dataset = OvercookedDataset(
            data_path=data_path,
            state_encoder_type=hparams['model_type'],
            max_samples=None # Load everything
        )
        print(f"Full dataset loaded with {len(full_dataset)} transitions.")
        if not hasattr(full_dataset, 'episode_info') or not full_dataset.episode_info:
             print("Warning: Full dataset loaded but failed to compute episode info.")
             # Decide how to handle this - maybe proceed without actual reward calculation?
             # For now, we'll let get_actual_cumulative_reward handle the None case.
             pass
    except Exception as e:
        print(f"Error loading full dataset: {e}. Cannot calculate actual rewards accurately.")
        return # Exit if we can't get episode info

    # Determine the actual number of samples to evaluate
    num_eval_samples = min(num_samples, len(full_dataset))
    print(f"Evaluating on {num_eval_samples} starting states...")
    eval_indices = range(num_eval_samples) # Use these indices for sampling

    # --- Load Test Data (Subset for iteration - potentially remove later if not needed) ---
    # We still need the raw dataset access to get state dicts for the planner
    # and potentially sequences for actual reward calculation
    # print("Loading test dataset subset (if needed)...")
    # test_dataset = OvercookedDataset(
    #     data_path=data_path,
    #     state_encoder_type=hparams['model_type'],
    #     max_samples=num_samples # Load only the required number of start states
    # )
    # print(f"Subset dataset loaded with {len(test_dataset)} samples.")
    
    # Evaluate models
    dynamics_errors = []
    reward_errors = []
    planner_predicted_rewards = []
    actual_cumulative_rewards = [] # Placeholder
    
    # num_eval_samples = min(num_samples, len(test_dataset)) # Now based on full_dataset length
    # print(f"Evaluating on {num_eval_samples} starting states...")
    
    for i in eval_indices: # Iterate through the chosen indices
        print(f"Processing sample index {i} (step {i+1}/{num_eval_samples})")
        # Get the i-th transition from the dataset
        # state_tensor, action_tensor, next_state_tensor, reward_tensor = test_dataset[i]
        
        # We need the original state dict for the planner
        # This requires modifying OvercookedDataset or reloading the raw data here.
        # ---- TEMPORARY WORKAROUND: Reload raw data for the i-th sample ----
        try:
            # TODO: This is inefficient. Dataset should provide state_dict or index mapping.
            df = pd.read_csv(data_path, skiprows=range(1, i + 1), nrows=1) 
            if len(df) > 0:
                current_state_json = df.iloc[0]['state']
                current_state_dict = parse_state(current_state_json)
                # Also get actual reward for single-step comparison (optional)
                # actual_reward = float(df.iloc[0]['reward'])
            else:
                 print(f"Warning: Could not read sample {i} from data file.")
                 continue
        except Exception as e:
            print(f"Error reading sample state {i} from {data_path}: {e}")
            continue
        # ---- END WORKAROUND ----
        
        # --- Single-step model evaluation (Optional - keep if needed) --- 
        # state = state_tensor.to(device).unsqueeze(0)
        # action = action_tensor.to(device).unsqueeze(0)
        # next_state = next_state_tensor.to(device).unsqueeze(0)
        # reward = reward_tensor.to(device)
        # with torch.no_grad():
        #     pred_next_state = dynamics_model(state, action)
        #     pred_reward = reward_model(state, action).squeeze()
        #     if pred_next_state.shape != next_state.shape:
        #          pred_next_state = torch.nn.functional.interpolate(pred_next_state, size=next_state.shape[2:], mode='nearest')
        #     dynamics_error = torch.nn.functional.mse_loss(pred_next_state, next_state).item()
        #     reward_error = torch.abs(pred_reward - reward).item()
        #     dynamics_errors.append(dynamics_error)
        #     reward_errors.append(reward_error)
        # --- End single-step evaluation ---

        # --- Planner Evaluation --- 
        try:
            # Plan action and get predicted cumulative reward for the best sequence
            planned_action_indices, predicted_cumulative_reward = planner.plan(current_state_dict)
            planner_predicted_rewards.append(predicted_cumulative_reward)

            # Get actual cumulative reward from the full dataset instance
            # Use index `i` which corresponds to the row in the original CSV/full dataset
            actual_reward_h = get_actual_cumulative_reward(full_dataset, i, planning_horizon) 
            actual_cumulative_rewards.append(actual_reward_h)

        except Exception as e:
            print(f"Error during planning for sample {i}: {e}")
            # Append NaN or skip if planning fails
            planner_predicted_rewards.append(np.nan)
            actual_cumulative_rewards.append(np.nan)
            continue
    
    # --- Print Results --- 
    print("\n--- Evaluation Results ---")
    # Print single-step results if calculated
    if dynamics_errors and reward_errors:
         avg_dynamics_error = np.mean(dynamics_errors)
         avg_reward_error = np.mean(reward_errors)
         print(f"Single-Step Model Performance ({len(dynamics_errors)} samples):")
         print(f"  Average Dynamics MSE: {avg_dynamics_error:.6f}")
         print(f"  Average Reward MAE: {avg_reward_error:.6f}")

    # Print planner results
    valid_planner_rewards = [r for r in planner_predicted_rewards if not np.isnan(r)]
    valid_actual_rewards = [r for r in actual_cumulative_rewards if not np.isnan(r)]
    num_valid_planner_samples = len(valid_planner_rewards)
    
    if num_valid_planner_samples > 0:
        avg_planner_pred_reward = np.mean(valid_planner_rewards)
        avg_actual_reward = np.mean(valid_actual_rewards)
        # Calculate correlation or difference if needed
        reward_diff = np.mean(np.array(valid_planner_rewards) - np.array(valid_actual_rewards)) if len(valid_actual_rewards) == num_valid_planner_samples else np.nan

        print(f"\nPlanner Performance ({num_valid_planner_samples}/{num_eval_samples} valid samples):")
        print(f"  Average Predicted Cumulative Reward (H={planning_horizon}): {avg_planner_pred_reward:.4f}")
        print(f"  Average Actual Cumulative Reward (H={planning_horizon}): {avg_actual_reward:.4f}")
        print(f"  Average Difference (Predicted - Actual): {reward_diff:.4f}")
    else:
        print("\nPlanner Performance: No valid planner samples processed.")

    # --- Plotting (Optional) --- 
    # Plot error distributions or planner vs actual rewards if desired
    # plt.figure(figsize=(12, 5))
    # ... plotting code ...
    # plt.savefig(os.path.join(model_dir, 'planner_evaluation.png'))
    # print(f"Planner evaluation plot saved to {os.path.join(model_dir, 'planner_evaluation.png')}")


def main():
    parser = argparse.ArgumentParser(description="Test PLDM models for Overcooked-AI")
    
    # Config file parameters
    parser.add_argument("--config", type=str, 
                        help="Path to configuration file (YAML or JSON)")
    
    # Optional parameters to override config
    parser.add_argument("--data_path", type=str,
                        help="Path to CSV dataset (overrides config)")
    parser.add_argument("--model_dir", type=str,
                        help="Directory containing the trained models (overrides config)")
    parser.add_argument("--num_samples", type=int,
                        help="Number of samples to evaluate (overrides config's testing.num_samples)")
    parser.add_argument("--device", type=str,
                        help="Device to use for evaluation (overrides config's training.device)")
    # Planner specific arguments
    parser.add_argument("--planning_horizon", type=int,
                        help="Planning horizon for MPC evaluation (overrides config's testing.planning_horizon)")
    parser.add_argument("--planning_samples", type=int,
                        help="Number of action sequences to sample for planner evaluation (overrides config's testing.planning_samples)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Get default configuration
    config = get_default_config()
    
    # Load config file if provided
    if args.config:
        try:
            loaded_config = load_config(args.config)
            config = merge_configs(config, loaded_config)
        except FileNotFoundError:
             print(f"Error: Config file {args.config} not found. Using defaults.")
    
    # Override config with command-line arguments if they were provided
    override_config = {}
    if args.model_dir:
        override_config.setdefault("training", {})["output_dir"] = args.model_dir
    if args.data_path:
        override_config.setdefault("testing", {})["test_data_path"] = args.data_path
    if args.num_samples is not None:
        override_config.setdefault("testing", {})["num_samples"] = args.num_samples
    if args.device:
        override_config.setdefault("training", {})["device"] = args.device
    if args.planning_horizon is not None:
         override_config.setdefault("testing", {})["planning_horizon"] = args.planning_horizon
    if args.planning_samples is not None:
         override_config.setdefault("testing", {})["planning_samples"] = args.planning_samples
    if args.seed is not None:
        override_config.setdefault("testing", {})["seed"] = args.seed
         
    # Merge overrides
    config = merge_configs(config, override_config)
    
    # Determine model directory
    model_dir = config["training"]["output_dir"]
    
    # Determine data path for testing
    test_data_path = config["testing"].get("test_data_path", None) # Use .get for safety
    if test_data_path is None:
        test_data_path = config["data"]["train_data_path"]
    
    # Determine number of samples and planning parameters from final config
    num_samples = config["testing"]["num_samples"]
    planning_horizon = config["testing"]["planning_horizon"]
    planning_samples = config["testing"]["planning_samples"]
    
    # Determine device
    device_str = config["training"]["device"]
    if device_str == "auto":
        device = None  # Auto-detect in the evaluate_models function
    else:
        device = torch.device(device_str)
    
    # Get seed value from config or command-line arg (command-line has priority)
    seed = args.seed if args.seed is not None else config.get("seed")
    if seed is not None:
        print(f"Using seed from {'command-line' if args.seed is not None else 'config'}: {seed}")
    
    # Evaluate models, passing planner args from final config
    evaluate_models(
        data_path=test_data_path,
        model_dir=model_dir,
        num_samples=num_samples,
        planning_horizon=planning_horizon,
        planning_samples=planning_samples,
        device=device,
        seed=seed  # Pass seed to evaluation function
    )


if __name__ == "__main__":
    main() 