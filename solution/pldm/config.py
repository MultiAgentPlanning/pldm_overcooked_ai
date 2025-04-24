import os
import yaml
import json
from pathlib import Path
import copy
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config: Dictionary containing configuration parameters
        config_path: Path to save the configuration file
    """
    config_path = Path(config_path)
    os.makedirs(config_path.parent, exist_ok=True)
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.
    
    Returns:
        Dictionary containing default configuration parameters
    """
    return {
        "seed": 42,  # Random seed for reproducibility
        
        # Workflow options
        "workflow": {
            "run_training": True,
            "run_probing": True,
            "run_planning": True
        },
        
        # Data parameters
        "data": {
            "train_data_path": "./data/train.csv",
            "val_data_path": None,  # If None, val_ratio will be used
            "test_data_path": None,  # If None, test_ratio will be used
            "val_ratio": 0.1,  # Ratio of data for validation
            "test_ratio": 0.1,  # Ratio of data for testing
            "max_samples": None,  # Maximum number of samples to use (None for all)
            "grid_size": 5,  # Size of grid for grid-based representation
        },
        
        # Model parameters
        "model": {
            "type": "grid",  # Model type: 'grid' or 'vector'
            "state_embed_dim": 128,  # Dimension of state embedding
            "action_embed_dim": 4,  # Dimension of action embedding
            "num_actions": 6,  # Number of possible actions
            "dynamics_hidden_dim": 128,  # Hidden dimension for dynamics model
            "reward_hidden_dim": 64,  # Hidden dimension for reward model
            "grid_height": None,  # Grid height (None for auto-detect)
            "grid_width": None  # Grid width (None for auto-detect)
        },
        
        # Loss function parameters
        "loss": {
            "dynamics_loss": "mse",  # Loss type for dynamics: 'mse' or 'vicreg'
            "reward_loss": "mse",    # Loss type for reward: 'mse' or 'vicreg'
            
            # VICReg specific parameters (if using VICReg)
            "vicreg": {
                "projector_layers": [2048, 2048, 2048],  # MLP projector dimensions
                "output_dim": 256,                      # Projector output dimension
                "sim_coeff": 25.0,                      # Coefficient for similarity loss
                "std_coeff": 25.0,                      # Coefficient for std loss
                "cov_coeff": 1.0,                       # Coefficient for covariance loss
                "std_margin": 1.0                       # Margin for std loss
            }
        },
        
        # Training parameters
        "training": {
            "output_dir": "models/pldm",  # Directory to save models
            "batch_size": 32,  # Batch size
            "dynamics_epochs": 10,  # Number of epochs for dynamics model
            "reward_epochs": 5,  # Number of epochs for reward model
            "learning_rate": 0.001,  # Learning rate
            "log_interval": 10,  # Interval for logging
            "num_workers": 4,  # Number of workers for data loading
            "device": "auto"  # Device for training ('cuda', 'cpu', or 'auto')
        },
        
        # Testing parameters
        "testing": {
            "num_samples": 100,  # Number of samples for evaluation
            "planning_horizon": 5,  # Planning horizon
            "planning_samples": 100  # Number of samples for planning
        },
        
        # WandB parameters
        "wandb": {
            "use_wandb": False,  # Whether to use WandB
            "project": "pldm-overcooked",  # Project name
            "name": None,  # Run name (None for auto-generated)
            "entity": None  # WandB entity (None for default)
        },
        
        # Probing parameters
        "probing": {
            "targets": ["agent_pos", "reward"],  # Probing targets
            "probe_dynamics": True,  # Whether to probe dynamics model
            "probe_reward": True,  # Whether to probe reward model
            "batch_size": 64,  # Batch size for probing
            "epochs": 10,  # Number of epochs for probing
            "lr": 0.001,  # Learning rate for probing
            "val_ratio": 0.1,  # Ratio of data for validation
            "test_ratio": 0.1,  # Ratio of data for testing
            "visualize": True,  # Whether to visualize results
            "max_vis_samples": 10  # Maximum number of samples to visualize
        }
    }


def create_default_config_file(output_path: str) -> None:
    """
    Create a default configuration file.
    
    Args:
        output_path: Path to save the default configuration file
    """
    config = get_default_config()
    save_config(config, output_path)
    print(f"Default configuration saved to {output_path}")


def merge_configs(default_config: Dict[str, Any], 
                 override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge a default configuration with an override configuration.
    
    Args:
        default_config: Default configuration dictionary
        override_config: Configuration dictionary to override defaults
        
    Returns:
        Merged configuration dictionary
    """
    if override_config is None:
        return default_config.copy()
    
    merged_config = default_config.copy()
    
    for key, value in override_config.items():
        if key in merged_config and isinstance(value, dict) and isinstance(merged_config[key], dict):
            # Recursively merge nested dictionaries
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            # Override with new value
            merged_config[key] = value
    
    return merged_config


if __name__ == "__main__":
    # Create a default configuration file when run directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a default PLDM configuration file")
    parser.add_argument("--output", type=str, default="config.yaml",
                        help="Path to save the default configuration file")
    
    args = parser.parse_args()
    create_default_config_file(args.output) 