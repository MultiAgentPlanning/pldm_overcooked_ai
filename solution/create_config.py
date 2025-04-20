#!/usr/bin/env python3
"""
Script to create a default configuration file for PLDM training and testing.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the pldm module
sys.path.append(str(Path(__file__).parent.parent))

from solution.pldm.config import get_default_config, save_config


def main():
    parser = argparse.ArgumentParser(
        description="Create a default configuration file for PLDM training and testing"
    )
    
    parser.add_argument("--output", type=str, default="config.yaml",
                        help="Path to save the default configuration file")
    parser.add_argument("--data_path", type=str, 
                        default="/scratch/hs5580/ddrl/overcooked/data/2020_hh_trials.csv",
                        help="Path to the data CSV file")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/hs5580/ddrl/overcooked/models",
                        help="Directory to save models")
    parser.add_argument("--model_type", type=str, default="grid", choices=["grid", "vector"],
                        help="Type of model ('grid' or 'vector')")
    
    args = parser.parse_args()
    
    # Get default configuration
    config = get_default_config()
    
    # Update with provided paths
    config["data"]["train_data_path"] = args.data_path
    config["training"]["output_dir"] = args.output_dir
    config["model"]["type"] = args.model_type
    
    # Create the default configuration file
    save_config(config, args.output)
    print(f"Configuration file saved to {args.output}")


if __name__ == "__main__":
    main() 