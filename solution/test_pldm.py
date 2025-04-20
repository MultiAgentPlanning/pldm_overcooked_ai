import argparse
import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import the pldm module
sys.path.append(str(Path(__file__).parent.parent))

from solution.pldm.utils import parse_state, parse_joint_action, get_action_index
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


def evaluate_models(data_path, model_dir, num_samples=100, device=None):
    """
    Evaluate the dynamics and reward models on a subset of the data.
    
    Args:
        data_path: Path to the test data
        model_dir: Directory containing the trained models
        num_samples: Number of samples to evaluate
        device: Device to use for evaluation
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    state_encoder, dynamics_model, reward_model, hparams = load_models(model_dir)
    dynamics_model.to(device)
    reward_model.to(device)
    
    # Load test data
    test_dataset = OvercookedDataset(
        data_path=data_path,
        state_encoder_type=hparams['model_type'],
        max_samples=num_samples
    )
    
    # Create a custom collate function for evaluation batches
    from solution.pldm.data_processor import custom_collate_fn
    
    # Evaluate models
    dynamics_errors = []
    reward_errors = []
    
    print(f"Evaluating on {min(num_samples, len(test_dataset))} samples...")
    
    for i in range(min(num_samples, len(test_dataset))):
        state, action, next_state, reward = test_dataset[i]
        
        # Move tensors to device
        state = state.to(device).unsqueeze(0)  # Add batch dimension
        action = action.to(device).unsqueeze(0)
        next_state = next_state.to(device).unsqueeze(0)
        reward = reward.to(device)
        
        # Make predictions
        with torch.no_grad():
            try:
                pred_next_state = dynamics_model(state, action)
                pred_reward = reward_model(state, action).squeeze()
                
                # Make sure the prediction and target have the same shape for comparison
                if pred_next_state.shape != next_state.shape:
                    # Resize prediction to match target shape (using interpolation)
                    pred_next_state = torch.nn.functional.interpolate(
                        pred_next_state, 
                        size=next_state.shape[2:], 
                        mode='nearest'
                    )
                
                # Calculate errors
                dynamics_error = torch.nn.functional.mse_loss(pred_next_state, next_state).item()
                reward_error = torch.abs(pred_reward - reward).item()
                
                dynamics_errors.append(dynamics_error)
                reward_errors.append(reward_error)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    # Print results
    if len(dynamics_errors) > 0:
        avg_dynamics_error = np.mean(dynamics_errors)
        avg_reward_error = np.mean(reward_errors)
        
        print(f"Evaluation results on {len(dynamics_errors)} samples:")
        print(f"  Average dynamics MSE: {avg_dynamics_error:.6f}")
        print(f"  Average reward MAE: {avg_reward_error:.6f}")
        
        # Plot error distributions
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(dynamics_errors, bins=20)
        plt.xlabel('MSE')
        plt.ylabel('Count')
        plt.title('Dynamics Prediction Error')
        
        plt.subplot(1, 2, 2)
        plt.hist(reward_errors, bins=20)
        plt.xlabel('MAE')
        plt.ylabel('Count')
        plt.title('Reward Prediction Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'evaluation_errors.png'))
        print(f"Error histograms saved to {os.path.join(model_dir, 'evaluation_errors.png')}")
    else:
        print("No valid samples were processed. Unable to compute evaluation metrics.")


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
                        help="Number of samples to evaluate (overrides config)")
    parser.add_argument("--device", type=str,
                        help="Device to use for evaluation (overrides config)")
    
    args = parser.parse_args()
    
    # Get default configuration
    config = get_default_config()
    
    # Load config file if provided
    if args.config:
        loaded_config = load_config(args.config)
        config = merge_configs(config, loaded_config)
    
    # Override config with command-line arguments
    override_config = {}
    
    # Model directory
    if args.model_dir:
        if "training" not in override_config:
            override_config["training"] = {}
        override_config["training"]["output_dir"] = args.model_dir
    
    # Data path
    if args.data_path:
        if "testing" not in override_config:
            override_config["testing"] = {}
        override_config["testing"]["test_data_path"] = args.data_path
    
    # Number of samples
    if args.num_samples:
        if "testing" not in override_config:
            override_config["testing"] = {}
        override_config["testing"]["num_samples"] = args.num_samples
    
    # Device
    if args.device:
        if "training" not in override_config:
            override_config["training"] = {}
        override_config["training"]["device"] = args.device
    
    # Merge override config with loaded config
    config = merge_configs(config, override_config)
    
    # Determine model directory
    model_dir = config["training"]["output_dir"]
    
    # Determine data path for testing
    test_data_path = config["testing"]["test_data_path"]
    if test_data_path is None:
        test_data_path = config["data"]["train_data_path"]
    
    # Determine number of samples
    num_samples = config["testing"]["num_samples"]
    
    # Determine device
    device_str = config["training"]["device"]
    if device_str == "auto":
        device = None  # Auto-detect in the evaluate_models function
    else:
        device = torch.device(device_str)
    
    # Evaluate models
    evaluate_models(
        data_path=test_data_path,
        model_dir=model_dir,
        num_samples=num_samples,
        device=device
    )


if __name__ == "__main__":
    main() 