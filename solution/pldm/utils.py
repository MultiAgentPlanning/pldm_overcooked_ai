import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional
import logging
import sys

# Action mapping constants
STAY = [0, 0]
UP = [-1, 0]
DOWN = [1, 0]
LEFT = [0, -1]
RIGHT = [0, 1]
INTERACT = "interact"

# Action indices
ACTION_INDICES = {
    str(STAY): 0,
    str(UP): 1,
    str(DOWN): 2,
    str(LEFT): 3, 
    str(RIGHT): 4,
    INTERACT: 5
}

ACTION_INDICES_REVERSE = {
    0: STAY,
    1: UP,
    2: DOWN,
    3: LEFT,
    4: RIGHT,
    5: INTERACT
}

# Layout element mapping
LAYOUT_ELEMENTS = {
    'X': 'wall',
    'P': 'pot',
    'O': 'onion_dispenser',
    'T': 'tomato_dispenser',
    'D': 'dish_dispenser',
    'S': 'serving_location',
    ' ': 'counter'
}

def parse_state(state_json: str) -> Dict:
    """Parse a state JSON string into a Python dictionary."""
    return json.loads(state_json)

def parse_joint_action(action_json: str) -> List:
    """Parse a joint action JSON string into a Python list."""
    # Handle the special case of joint actions with "interact"
    if '"interact"' in action_json:
        # Replace the "interact" string to make it easier to parse
        action_json = action_json.replace('"interact"', '"INTERACT"')
        action_list = json.loads(action_json)
        # Convert back "INTERACT" to the string "interact"
        action_list = [a if a != "INTERACT" else INTERACT for a in action_list]
        return action_list
    else:
        # Standard json parsing for movement actions
        return json.loads(action_json)

def get_action_index(action) -> int:
    """Convert an action to its corresponding index."""
    if isinstance(action, str) and action == INTERACT:
        return ACTION_INDICES[INTERACT]
    else:
        return ACTION_INDICES[str(action)]

def parse_layout(layout_json: str) -> List[List[str]]:
    """Parse a layout JSON string into a 2D grid."""
    return json.loads(layout_json)

def encode_layout(layout_grid: List[List[str]]) -> np.ndarray:
    """
    Encode layout grid into one-hot channels for each element type.
    Returns a tensor of shape (num_element_types, height, width)
    """
    height = len(layout_grid)
    width = len(layout_grid[0])
    num_elements = len(LAYOUT_ELEMENTS)
    
    # Create a tensor to hold the one-hot encoding
    layout_tensor = np.zeros((num_elements, height, width))
    
    # Fill in the tensor based on the layout grid
    for i, row in enumerate(layout_grid):
        for j, cell in enumerate(row):
            if cell in LAYOUT_ELEMENTS:
                element_idx = list(LAYOUT_ELEMENTS.keys()).index(cell)
                layout_tensor[element_idx, i, j] = 1
                
    return layout_tensor

def get_one_hot(idx: int, size: int) -> np.ndarray:
    """Create a one-hot vector with 1 at the specified index."""
    one_hot = np.zeros(size)
    one_hot[idx] = 1
    return one_hot

# Add the reverse mapping: index to action name
INDEX_TO_ACTION = {
    0: "STAY",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "INTERACT"
}

def index_to_action_name(action_idx):
    """Convert action index to human-readable name."""
    action_names = ['STAY', 'UP', 'RIGHT', 'DOWN', 'LEFT', 'INTERACT']
    if 0 <= action_idx < len(action_names):
        return action_names[action_idx]
    return f"UNKNOWN({action_idx})"

def setup_logger(level=logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configures the root logger.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file to save logs.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplication if called multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
        
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up log file at {log_file}: {e}")

    logger.info("Logger configured.")

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Integer seed value to use
    """
    import random
    import numpy as np
    import torch
    
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seeds
    torch.manual_seed(seed)
    
    # Set CUDA's random seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        
        # Additional settings for deterministic CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Return seed in case it's auto-generated in the future
    return seed 