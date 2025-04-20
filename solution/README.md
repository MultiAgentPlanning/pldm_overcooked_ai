# PLDM for Overcooked-AI

This repository contains an implementation of Predictive Latent Dynamics Models (PLDM) for the Overcooked-AI environment. The implementation includes:

1. **State Encoder**: Converts Overcooked game states into grid-based or vector-based representations
2. **Dynamics Predictor**: Predicts the next state given the current state and joint action
3. **Reward Predictor**: Predicts the reward for a given state and action
4. **Training & Evaluation**: Tools for training and evaluating the models

## Latest Updates

### Dynamic Model Architecture (2023)

The PLDM implementation now supports dynamic model architectures that can adapt to different grid sizes:

- **Adaptive Neural Networks**: Models now dynamically adjust to varying input dimensions, without requiring fixed grid sizes.
- **Runtime Layer Creation**: Linear layers are created on-demand based on actual input shapes.
- **Flexible Data Processing**: Improved data handling with custom collation for tensors of different sizes.
- **Type Conversion**: Automatic conversion of configuration parameters to appropriate types.

These improvements allow the models to:
- Work with variable sized grids
- Handle inconsistent state dimensions across batches
- Adapt to different data formats without manual intervention

## Configuration System

The PLDM implementation uses a configuration system to manage training and testing parameters. Configuration files can be in YAML or JSON format.

### Default Configuration Files

Use one of the provided sample configurations:
- `configs/default_grid_config.yaml`: Default configuration for grid-based models
- `configs/vector_model_config.yaml`: Configuration for vector-based models
- `configs/testing_config.yaml`: Configuration for quick testing with reduced dataset size

### Configuration Structure

The configuration file has the following sections:

1. **data**: Parameters related to the dataset
   - `train_data_path`: Path to the training data CSV
   - `val_data_path`: Path to validation data (if null, uses a split of training data)
   - `val_ratio`: Ratio of training data to use for validation
   - `max_samples`: Maximum number of samples to use (for testing)

2. **model**: Parameters for the model architecture
   - `type`: Type of state encoding ('grid' or 'vector')
   - `state_embed_dim`: Dimension of state embedding
   - `action_embed_dim`: Dimension of action embedding
   - `num_actions`: Number of possible actions
   - `dynamics_hidden_dim`: Hidden dimension for the dynamics predictor
   - `reward_hidden_dim`: Hidden dimension for the reward predictor
   - `grid_height`: Height of the grid (optional, can be determined from data)
   - `grid_width`: Width of the grid (optional, can be determined from data)

3. **training**: Parameters for training
   - `output_dir`: Directory to save models and logs
   - `batch_size`: Batch size for training
   - `learning_rate`: Learning rate
   - `dynamics_epochs`: Number of epochs to train the dynamics model
   - `reward_epochs`: Number of epochs to train the reward model
   - `log_interval`: Interval for logging training progress
   - `num_workers`: Number of workers for the dataloader
   - `device`: Device to use for training ('auto', 'cuda', or 'cpu')

4. **testing**: Parameters for testing
   - `test_data_path`: Path to test data (if null, uses training data)
   - `num_samples`: Number of samples to evaluate

## Training Models

To train a model using a configuration file:

```bash
python solution/train_pldm.py --config /scratch/hs5580/ddrl/overcooked/solution/configs/default_grid_config.yaml
```

You can override configuration parameters using command-line arguments:

```bash
python solution/train_pldm.py --config configs/default_grid_config.yaml --batch_size 32 --device cuda
```

For quick testing, you can use the testing configuration:

```bash
python solution/train_pldm.py --config configs/testing_config.yaml
```

Or use the `--test` flag with any configuration to use a small subset of data:

```bash
python solution/train_pldm.py --config configs/default_grid_config.yaml --test
```

### Using Singularity Container

If running with a Singularity container:

```bash
singularity exec --nv --bind /path/to/overcooked:/path/to/overcooked /path/to/container.sif python solution/train_pldm.py --config /path/to/config.yaml --test
```

## Testing Models

To test a trained model:

```bash
python solution/test_pldm.py --config configs/default_grid_config.yaml
```

You can also override configuration parameters:

```bash
python solution/test_pldm.py --config configs/default_grid_config.yaml --model_dir ./models/my_model --num_samples 200
```

## Handling Different Grid Sizes

The updated implementation automatically handles different grid sizes through:

1. **Custom Collation**: The data loader uses a custom collate function that pads tensors to the maximum size in each batch.
2. **Dynamic Layer Creation**: The models create projection layers on-demand based on the actual input dimensions.
3. **Dimension Tracking**: The models track input and output dimensions and adjust as needed during training.

This means you don't need to worry about specifying exact grid dimensions in the configuration - they can be determined from the data.

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- PyYAML

## Directory Structure

```
solution/
├── configs/                    # Sample configuration files
│   ├── default_grid_config.yaml
│   ├── vector_model_config.yaml
│   └── testing_config.yaml
├── pldm/                       # PLDM implementation
│   ├── __init__.py
│   ├── config.py              # Configuration utilities
│   ├── utils.py               # Utility functions
│   ├── state_encoder.py       # State encoding classes
│   ├── dynamics_predictor.py  # Dynamics prediction models
│   ├── reward_predictor.py    # Reward prediction models
│   ├── data_processor.py      # Data loading and processing
│   └── trainer.py             # Training utilities
├── create_config.py            # Script to create configuration files
├── train_pldm.py               # Script for training models
├── test_pldm.py                # Script for testing models
└── README.md                   # This file
```

## Troubleshooting

### Common Issues

1. **Dimension Mismatch Errors**: If you encounter errors about matrix dimensions not matching, it's likely because:
   - The model is trying to process states with different grid sizes
   - There are inconsistencies in the data
   - Solution: The new dynamic architecture should handle this, but you can also try setting explicit grid dimensions in the config.

2. **CUDA Out of Memory**: If you run out of GPU memory:
   - Reduce the batch size
   - Use a smaller model (reduce hidden dimensions)
   - Use the CPU instead of GPU for very large models

3. **Slow Training**: If training is too slow:
   - Increase the number of workers in the configuration
   - Use a smaller subset of data for testing with `max_samples`
   - Reduce the number of epochs
   - Try using the vector model which is typically faster than the grid model

### Debug Output

The models now print helpful debug information during training:
- When new layers are created due to dimension changes
- The dimensions of the data being processed
- Total number of parameters in each model 