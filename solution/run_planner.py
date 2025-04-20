import argparse
import os
import sys
import torch
import json
from pathlib import Path

# Add the parent directory to the path so we can import the solution module
sys.path.append(str(Path(__file__).parent.parent))

from solution.pldm import (
    Planner,
    GridStateEncoder, VectorStateEncoder,
    GridDynamicsPredictor, VectorDynamicsPredictor,
    GridRewardPredictor, VectorRewardPredictor,
    load_config, get_default_config, merge_configs
)
from solution.test_pldm import load_models # Reuse model loading logic
from solution.pldm.utils import parse_state, index_to_action_name, set_seeds


def get_sample_state(data_path):
    """Loads the first state from the dataset for testing."""
    import pandas as pd
    try:
        df = pd.read_csv(data_path, nrows=1)
        if len(df) > 0:
            state_json = df.iloc[0]['state']
            return parse_state(state_json)
        else:
            print("Warning: Could not read any rows from the data file.")
            return None
    except Exception as e:
        print(f"Error reading sample state from {data_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run PLDM Planner for Overcooked-AI")

    # Config file parameters
    parser.add_argument("--config", type=str, 
                        help="Path to configuration file (YAML or JSON) used for training the models")
    
    # Optional parameters to override config
    parser.add_argument("--model_dir", type=str,
                        help="Directory containing the trained models (overrides config)")
    parser.add_argument("--horizon", type=int, default=10,
                        help="Planning horizon for MPC")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of action sequences to sample for planning")
    parser.add_argument("--device", type=str,
                        help="Device to use for planning ('auto', 'cuda', 'cpu', overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible action sampling")

    args = parser.parse_args()

    # --- 1. Load Configuration --- 
    config = get_default_config()
    if args.config:
        try:
            loaded_config = load_config(args.config)
            config = merge_configs(config, loaded_config)
            print(f"Loaded configuration from {args.config}")
        except FileNotFoundError:
            print(f"Error: Config file not found at {args.config}. Using default config.")
    
    # Override specific config values if provided via args
    if args.model_dir:
        config["training"]["output_dir"] = args.model_dir
    if args.device:
         config["training"]["device"] = args.device # Use training device setting for consistency

    model_dir = config["training"]["output_dir"]
    device_str = config["training"]["device"]
    data_path = config["data"]["train_data_path"]
    
    # --- 2. Determine Device --- 
    if device_str == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # --- Set Random Seed if Provided ---
    # Check config first, but command-line arg has priority
    seed = args.seed if args.seed is not None else config.get("seed")
    if seed is not None:
        print(f"Setting random seed to {seed} (from {'command-line' if args.seed is not None else 'config'})")
        set_seeds(seed)

    # --- 3. Load Models --- 
    print(f"Loading models from: {model_dir}")
    try:
        state_encoder, dynamics_model, reward_model, hparams = load_models(model_dir)
        print("Models loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model files not found in {model_dir}. Make sure models are trained.")
        return
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # --- 4. Initialize Planner --- 
    num_actions = hparams['num_actions'] # Get num_actions from loaded hyperparameters
    planner = Planner(
        dynamics_model=dynamics_model,
        reward_model=reward_model,
        state_encoder=state_encoder,
        num_actions=num_actions,
        planning_horizon=args.horizon,
        num_samples=args.samples,
        device=device,
        seed=seed
    )

    # --- 5. Get a Sample State --- 
    print(f"Loading sample state from: {data_path}")
    current_state = get_sample_state(data_path)
    if current_state is None:
        print("Could not get a sample state to plan from.")
        return
        
    print("\nSampled Initial State:")
    print(json.dumps(current_state, indent=2))

    # --- 6. Plan Action --- 
    print(f"\nPlanning action with horizon={args.horizon}, samples={args.samples}...")
    best_joint_action_indices = planner.plan(current_state)
    
    # Convert indices to action names for readability
    action_agent0 = index_to_action_name(best_joint_action_indices[0])
    action_agent1 = index_to_action_name(best_joint_action_indices[1])
    
    print(f"\nBest Joint Action Found:")
    print(f"  Agent 0: {action_agent0} (Index: {best_joint_action_indices[0]}) ")
    print(f"  Agent 1: {action_agent1} (Index: {best_joint_action_indices[1]}) ")

if __name__ == "__main__":
    main() 