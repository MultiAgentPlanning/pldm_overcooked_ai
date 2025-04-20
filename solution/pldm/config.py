import os
import yaml
import json
from pathlib import Path
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
    Get default configuration parameters for PLDM models.
    
    Returns:
        Dictionary containing default configuration parameters
    """
    return {
        # Data parameters
        "data": {
            "train_data_path": "../data/2020_hh_trials.csv",
            "val_data_path": None,  # If None, will use a split of train data
            "val_ratio": 0.1,       # Ratio of train data to use for validation if val_data_path is None
            "max_samples": None,    # Maximum number of samples to use (for testing)
        },
        
        # Model parameters
        "model": {
            "type": "grid",         # 'grid' or 'vector'
            "state_embed_dim": 128,
            "action_embed_dim": 4,
            "num_actions": 6,
            "dynamics_hidden_dim": 256,
            "reward_hidden_dim": 64,
            "grid_height": 5,
            "grid_width": 13,
        },
        
        # Training parameters
        "training": {
            "output_dir": "./models",
            "batch_size": 64,
            "learning_rate": 1e-4,
            "dynamics_epochs": 10,
            "reward_epochs": 5,
            "log_interval": 10,
            "num_workers": 4,
            "device": "auto",       # 'auto', 'cuda', or 'cpu'
        },
        
        # Testing parameters
        "testing": {
            "test_data_path": None,  # If None, will use training data
            "num_samples": 100,      # Number of samples to evaluate
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