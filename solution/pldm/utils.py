import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional

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