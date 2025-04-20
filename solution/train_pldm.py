import argparse
import os
import sys
from pathlib import Path

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
    
    args = parser.parse_args()
    
    # Create default config file if requested
    if args.create_config:
        create_default_config_file(args.create_config)
        return
    
    # Get default configuration
    config = get_default_config()
    
    # Load config file if provided
    if args.config:
        loaded_config = load_config(args.config)
        config = merge_configs(config, loaded_config)
    
    # Override config with command-line arguments
    override_config = {}
    
    # Data parameters
    if args.data_path:
        if "data" not in override_config:
            override_config["data"] = {}
        override_config["data"]["train_data_path"] = args.data_path
    
    # Model parameters
    if args.model_type:
        if "model" not in override_config:
            override_config["model"] = {}
        override_config["model"]["type"] = args.model_type
    
    # Training parameters
    if args.output_dir:
        if "training" not in override_config:
            override_config["training"] = {}
        override_config["training"]["output_dir"] = args.output_dir
    
    if args.batch_size:
        if "training" not in override_config:
            override_config["training"] = {}
        override_config["training"]["batch_size"] = args.batch_size
    
    if args.device:
        if "training" not in override_config:
            override_config["training"] = {}
        override_config["training"]["device"] = args.device
    
    # Testing parameters
    if args.max_samples:
        if "data" not in override_config:
            override_config["data"] = {}
        override_config["data"]["max_samples"] = args.max_samples
    
    # Apply test settings if --test flag is set
    if args.test:
        if "data" not in override_config:
            override_config["data"] = {}
        override_config["data"]["max_samples"] = 1000 if not args.max_samples else args.max_samples
        
        if "training" not in override_config:
            override_config["training"] = {}
        override_config["training"]["dynamics_epochs"] = 2
        override_config["training"]["reward_epochs"] = 2
        override_config["training"]["log_interval"] = 5
    
    # Merge override config with loaded config
    config = merge_configs(config, override_config)
    
    # Create output directory and save the used configuration
    output_dir = Path(config["training"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    save_config(config, output_dir / "training_config.yaml")
    
    # Determine device
    device = config["training"]["device"]
    if device == "auto":
        device = None  # Let the trainer auto-detect

    # Initialize trainer with configuration
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
        grid_height=config["model"]["grid_height"],
        grid_width=config["model"]["grid_width"],
        device=device,
        max_samples=config["data"]["max_samples"],
        num_workers=config["training"]["num_workers"]
    )
    
    # Train models
    trainer.train_all(
        dynamics_epochs=config["training"]["dynamics_epochs"],
        reward_epochs=config["training"]["reward_epochs"],
        log_interval=config["training"]["log_interval"]
    )
    
    print("Training complete. Models saved to", config["training"]["output_dir"])


if __name__ == "__main__":
    main() 