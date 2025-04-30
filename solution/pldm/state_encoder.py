import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional

from .utils import parse_state, encode_layout, get_one_hot

class GridStateEncoder:
    """
    Encodes the Overcooked state into a grid-based representation.
    The grid has multiple channels, each representing different features.
    """
    def __init__(self, grid_height: int = None, grid_width: int = None):
        # Set grid dimensions (can be None for dynamic determination)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_initialized = False if grid_height is None or grid_width is None else True
        
        # Define channels for the grid
        self.channels = {
            "agent1_position": 0,
            "agent1_orientation_up": 1,
            "agent1_orientation_down": 2,
            "agent1_orientation_left": 3,
            "agent1_orientation_right": 4,
            "agent2_position": 5,
            "agent2_orientation_up": 6,
            "agent2_orientation_down": 7,
            "agent2_orientation_left": 8,
            "agent2_orientation_right": 9,
            "agent1_holding_onion": 10,
            "agent1_holding_tomato": 11,
            "agent1_holding_dish": 12,
            "agent1_holding_soup": 13,
            "agent2_holding_onion": 14,
            "agent2_holding_tomato": 15,
            "agent2_holding_dish": 16,
            "agent2_holding_soup": 17,
            "onion_on_counter": 18,
            "tomato_on_counter": 19,
            "dish_on_counter": 20,
            "soup_on_counter": 21,
            "pot_empty": 22,
            "pot_cooking": 23,
            "pot_ready": 24,
            # Layout elements
            "wall": 25,
            "counter": 26,
            "onion_dispenser": 27,
            "tomato_dispenser": 28,
            "dish_dispenser": 29,
            "serving_location": 30,
            # Time information
            "timestep": 31
        }
        
        # Total number of channels
        self.num_channels = len(self.channels)
        
        # Dictionary for object type mapping
        self.object_types = {
            "onion": 0,
            "tomato": 1,
            "dish": 2,
            "soup": 3
        }
        
        # Dictionary for orientation mapping
        self.orientation_map = {
            (-1, 0): "up",    # Up
            (1, 0): "down",   # Down
            (0, -1): "left",  # Left
            (0, 1): "right"   # Right
        }
        
        # Dictionary for layout element mapping
        self.layout_elements = {
            'X': 'wall',
            'P': 'pot',
            'O': 'onion_dispenser',
            'T': 'tomato_dispenser',
            'D': 'dish_dispenser',
            'S': 'serving_location',
            ' ': 'counter'
        }
        
        # Mapping from layout element to channel index
        self.layout_element_to_channel = {
            'wall': self.channels["wall"],
            'counter': self.channels["counter"],
            'onion_dispenser': self.channels["onion_dispenser"],
            'tomato_dispenser': self.channels["tomato_dispenser"],
            'dish_dispenser': self.channels["dish_dispenser"],
            'serving_location': self.channels["serving_location"]
        }
        
        # Maximum timestep for normalization (can be adjusted based on data)
        self.max_timestep = 400  # checked this in csv.
    
    def _initialize_grid_size(self, state_dict: Dict):
        """
        Determine the grid dimensions by examining the positions in the state.
        
        Args:
            state_dict: A dictionary containing the game state.
        """
        # Initialize with minimum dimensions
        max_row = 0
        max_col = 0
        
        # Check player positions
        for player in state_dict["players"]:
            row, col = player["position"]
            max_row = max(max_row, row)
            max_col = max(max_col, col)
        
        # Check object positions
        for obj in state_dict.get("objects", []):
            if "position" in obj:
                row, col = obj["position"]
                max_row = max(max_row, row)
                max_col = max(max_col, col)
        
        # Check layout dimensions if available
        if "layout" in state_dict:
            layout = state_dict["layout"]
            max_row = max(max_row, len(layout) - 1)
            if layout and len(layout) > 0:
                max_col = max(max_col, len(layout[0]) - 1)
        
        # Set grid dimensions with margin
        self.grid_height = max_row + 2  # Add margin
        self.grid_width = max_col + 2   # Add margin
        print(f"Detected grid dimensions: {self.grid_height}x{self.grid_width}")
        self.grid_initialized = True
    
    def encode(self, state_dict: Dict) -> np.ndarray:
        """
        Encode the game state as a multi-channel grid representation.
        
        Args:
            state_dict: A dictionary containing the game state.
            
        Returns:
            A numpy array of shape (num_channels, grid_height, grid_width)
        """
        # Initialize grid size if not already done
        if not self.grid_initialized:
            self._initialize_grid_size(state_dict)
        
        # Further check for valid positions in this specific state
        for player in state_dict["players"]:
            row, col = player["position"]
            if row >= self.grid_height or col >= self.grid_width:
                # Expand grid if needed
                self.grid_height = max(self.grid_height, row + 1)
                self.grid_width = max(self.grid_width, col + 1)
        
        for obj in state_dict.get("objects", []):
            if "position" in obj:
                row, col = obj["position"]
                if row >= self.grid_height or col >= self.grid_width:
                    # Expand grid if needed
                    self.grid_height = max(self.grid_height, row + 1)
                    self.grid_width = max(self.grid_width, col + 1)
        
        # Check layout dimensions if available
        if "layout" in state_dict:
            layout = state_dict["layout"]
            if layout and len(layout) > 0:
                if len(layout) > self.grid_height or len(layout[0]) > self.grid_width:
                    self.grid_height = max(self.grid_height, len(layout))
                    self.grid_width = max(self.grid_width, len(layout[0]))
        
        # Initialize grid with zeros
        grid = np.zeros((self.num_channels, self.grid_height, self.grid_width))
        
        # Add timestep information (normalized between 0 and 1)
        timestep = state_dict.get("timestep", 0)
        normalized_timestep = min(timestep / self.max_timestep, 1.0)  # Cap at 1.0
        
        # Fill the entire timestep channel with the normalized value
        grid[self.channels["timestep"]] = normalized_timestep
        
        # Encode layout if available
        if "layout" in state_dict:
            layout = state_dict["layout"]
            for i, row in enumerate(layout):
                for j, cell in enumerate(row):
                    if cell in self.layout_elements:
                        element_name = self.layout_elements[cell]
                        if element_name in self.layout_element_to_channel:
                            grid[self.layout_element_to_channel[element_name], i, j] = 1
        
        # Encode players (agents)
        for player_idx, player in enumerate(state_dict["players"]):
            # Get player position
            row, col = player["position"]
            
            # Set agent position
            if player_idx == 0:  # First agent
                grid[self.channels["agent1_position"], row, col] = 1
            else:  # Second agent
                grid[self.channels["agent2_position"], row, col] = 1
            
            # Encode orientation
            orientation = tuple(player["orientation"])
            orient_name = self.orientation_map.get(orientation)
            
            if orient_name:
                if player_idx == 0:
                    grid[self.channels[f"agent1_orientation_{orient_name}"], row, col] = 1
                else:
                    grid[self.channels[f"agent2_orientation_{orient_name}"], row, col] = 1
            
            # Encode held object
            held_obj = player["held_object"]
            if held_obj:
                obj_name = held_obj.get("name", "")
                if obj_name in self.object_types:
                    if player_idx == 0:
                        grid[self.channels[f"agent1_holding_{obj_name}"], row, col] = 1
                    else:
                        grid[self.channels[f"agent2_holding_{obj_name}"], row, col] = 1
        
        # Encode objects in the environment
        for obj in state_dict.get("objects", []):
            obj_name = obj.get("name", "")
            obj_pos = obj.get("position", [0, 0])
            row, col = obj_pos
            
            if obj_name in ["onion", "tomato", "dish"]:
                grid[self.channels[f"{obj_name}_on_counter"], row, col] = 1
            elif obj_name == "soup":
                grid[self.channels["soup_on_counter"], row, col] = 1
            elif obj_name == "pot":
                # Check pot status
                if "cooking_tick" in obj and obj["cooking_tick"] > 0:
                    grid[self.channels["pot_cooking"], row, col] = 1
                elif "is_ready" in obj and obj["is_ready"]:
                    grid[self.channels["pot_ready"], row, col] = 1
                else:
                    grid[self.channels["pot_empty"], row, col] = 1
        
        return grid


class StateEncoderNetwork(nn.Module):
    """
    Neural network for encoding Overcooked states.
    Takes a grid-based state representation and embeds it into a latent vector.
    """
    def __init__(self, 
                 input_channels: int = 32, 
                 state_embed_dim: int = 128,
                 grid_height: int = None,
                 grid_width: int = None):
        super().__init__()
        
        self.input_channels = input_channels
        self.state_embed_dim = state_embed_dim
        
        # Grid dimensions will be set when first batch is passed
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.dynamic_net = True if grid_height is None or grid_width is None else False
        self.initialized = False
        
        if not self.dynamic_net:
            self._build_network()
    
    def _build_network(self):
        """Build the network with known grid dimensions."""
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = 64 * self.grid_height * self.grid_width
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, self.state_embed_dim)
        
        self.initialized = True
    
    def forward(self, x):
        """Forward pass through the network."""
        # If dynamic, initialize network on first forward pass with batch shape
        if self.dynamic_net and not self.initialized:
            _, _, self.grid_height, self.grid_width = x.shape
            self._build_network()
        
        # Input shape: [batch_size, channels, height, width]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation on final layer
        
        return x


class VectorStateEncoder:
    """
    Encodes Overcooked state as a flat feature vector.
    Useful for MLP-based models without convolutional layers.
    """
    def __init__(self, grid_height: int = None, grid_width: int = None):
        # Grid dimensions are only used for normalization
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_initialized = False
        
        # Dictionary for object type mapping
        self.object_types = {
            "onion": 0,
            "tomato": 1,
            "dish": 2,
            "soup": 3
        }
        
        # Dictionary for orientation mapping
        self.orientations = {
            (-1, 0): 0,  # Up
            (1, 0): 1,   # Down
            (0, -1): 2,  # Left
            (0, 1): 3    # Right
        }
        
        # Dictionary for layout element mapping
        self.layout_elements = {
            'X': 'wall',
            'P': 'pot',
            'O': 'onion_dispenser',
            'T': 'tomato_dispenser',
            'D': 'dish_dispenser',
            'S': 'serving_location',
            ' ': 'counter'
        }
        
        # Maximum timestep for normalization (can be adjusted based on data)
        self.max_timestep = 600  # Typical episode length in Overcooked
    
    def _initialize_grid_size(self, state_dict: Dict):
        """
        Determine the grid dimensions by examining the positions in the state.
        
        Args:
            state_dict: A dictionary containing the game state.
        """
        # Initialize with minimum dimensions
        max_row = 0
        max_col = 0
        
        # Check player positions
        for player in state_dict["players"]:
            row, col = player["position"]
            max_row = max(max_row, row)
            max_col = max(max_col, col)
        
        # Check object positions
        for obj in state_dict.get("objects", []):
            if "position" in obj:
                row, col = obj["position"]
                max_row = max(max_row, row)
                max_col = max(max_col, col)
        
        # Check layout dimensions if available
        if "layout" in state_dict:
            layout = state_dict["layout"]
            max_row = max(max_row, len(layout) - 1)
            if layout and len(layout) > 0:
                max_col = max(max_col, len(layout[0]) - 1)
        
        # Set grid dimensions with margin
        self.grid_height = max_row + 2  # Add margin
        self.grid_width = max_col + 2   # Add margin
        print(f"Detected grid dimensions for vector encoding: {self.grid_height}x{self.grid_width}")
        self.grid_initialized = True
        
    def encode(self, state_dict: Dict) -> np.ndarray:
        """
        Encode the game state as a flat feature vector.
        
        Args:
            state_dict: A dictionary containing the game state.
            
        Returns:
            A numpy array representing the state as a flat vector.
        """
        # Initialize grid size if needed for normalization
        if not self.grid_initialized:
            self._initialize_grid_size(state_dict)
        
        # Initialize vectors for agents
        features = []
        
        # Add timestep information (normalized between 0 and 1)
        timestep = state_dict.get("timestep", 0)
        normalized_timestep = min(timestep / self.max_timestep, 1.0)  # Cap at 1.0
        features.append(normalized_timestep)
        
        # Encode agents
        for player in state_dict["players"]:
            # Normalize positions to [0, 1]
            pos = player["position"]
            pos_norm = [pos[0]/self.grid_height, pos[1]/self.grid_width]
            features.extend(pos_norm)
            
            # Encode orientation as one-hot
            orientation = tuple(player["orientation"])
            orient_idx = self.orientations.get(orientation, 0)
            orient_onehot = get_one_hot(orient_idx, 4)
            features.extend(orient_onehot)
            
            # Encode held object as one-hot
            held_obj = player["held_object"]
            if held_obj and "name" in held_obj:
                obj_idx = self.object_types.get(held_obj["name"], 0)
                obj_onehot = get_one_hot(obj_idx, 5)  # 4 object types + 1 for "none"
                obj_onehot[obj_idx] = 1
            else:
                obj_onehot = get_one_hot(4, 5)  # "none" is index 4
            features.extend(obj_onehot)
        
        # Encode objects in the environment (simplified)
        # Count objects of each type
        obj_counts = {obj_type: 0 for obj_type in self.object_types}
        # Count pot states
        pot_states = {"empty": 0, "cooking": 0, "ready": 0}
        
        for obj in state_dict.get("objects", []):
            obj_name = obj.get("name", "")
            
            if obj_name in self.object_types:
                obj_counts[obj_name] += 1
            elif obj_name == "pot":
                if "cooking_tick" in obj and obj["cooking_tick"] > 0:
                    pot_states["cooking"] += 1
                elif "is_ready" in obj and obj["is_ready"]:
                    pot_states["ready"] += 1
                else:
                    pot_states["empty"] += 1
        
        # Add object counts to features
        features.extend(list(obj_counts.values()))
        # Add pot states to features
        features.extend(list(pot_states.values()))
        
        # Encode layout information if available (summarized)
        layout_counts = {element: 0 for element in self.layout_elements.values()}
        
        if "layout" in state_dict:
            layout = state_dict["layout"]
            for row in layout:
                for cell in row:
                    if cell in self.layout_elements:
                        element = self.layout_elements[cell]
                        layout_counts[element] += 1
        
        # Add layout counts to features
        features.extend(list(layout_counts.values()))
        
        return np.array(features, dtype=np.float32)


class VectorEncoderNetwork(nn.Module):
    """
    Neural network for encoding state vectors.
    Takes a flat vector state representation and embeds it.
    """
    def __init__(self, input_dim: int, state_embed_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.state_embed_dim = state_embed_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_embed_dim)
        
    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on final layer
        
        return x


class StateEncoder:
    """
    Encoder for Overcooked state representations.
    Converts game state to a tensor representation suitable for neural networks.
    Handles dynamic grid sizes based on environment data.
    """
    def __init__(self, env=None, grid_height=None, grid_width=None):
        """
        Initialize the state encoder with either an environment or explicit grid dimensions.
        
        Args:
            env: Overcooked environment (optional)
            grid_height: Grid height to use if env not provided (optional)
            grid_width: Grid width to use if env not provided (optional)
        """
        # Channel definitions for state encoding
        self.channel_idx = {
            "player_pos": 0,
            "player_orientation": 1,
            "counter": 2,
            "onion_supply": 3,
            "dish_supply": 4,
            "serve_area": 5,
            "onion": 6,
            "dish": 7,
            "soup": 8,
            "pot": 9,
        }
        self.num_channels = len(self.channel_idx)
        
        # Try to get grid dimensions from the environment
        if env is not None:
            # Extract grid dimensions from environment
            self.grid_height = env.mdp.state_space[0]
            self.grid_width = env.mdp.state_space[1]
        else:
            # Use provided grid dimensions or fallback to defaults
            self.grid_height = grid_height if grid_height is not None else 5
            self.grid_width = grid_width if grid_width is not None else 13
            
        # After detecting grid dimensions, print them for debugging
        print(f"Initialized StateEncoder with grid size: {self.grid_height}x{self.grid_width}")

    def encode_states(self, states):
        """
        Encode multiple game states into tensor representations.
        
        Args:
            states: List of game states
            
        Returns:
            Tensor of encoded states [batch_size, num_channels, grid_height, grid_width]
        """
        encoded_states = []
        for state in states:
            encoded_states.append(self.encode_state(state))
        return torch.stack(encoded_states)

    def encode_state(self, state):
        """
        Encode a single game state into a tensor representation.
        
        Args:
            state: A game state
            
        Returns:
            Tensor of encoded state [num_channels, grid_height, grid_width]
        """
        # Initialize empty grid representation
        grid = torch.zeros(self.num_channels, self.grid_height, self.grid_width)
        
        # Extract data from the state
        players = state["players"]
        objects = state["objects"]
        terrain = state["terrain"]
        
        # Iterate over the terrain grid to set up static environment elements
        for y in range(len(terrain)):
            for x in range(len(terrain[y])):
                if y >= self.grid_height or x >= self.grid_width:
                    continue  # Skip if outside of our grid
                    
                cell = terrain[y][x]
                
                if cell == "X":
                    # Counter
                    grid[self.channel_idx["counter"], y, x] = 1
                elif cell == "O":
                    # Onion supply
                    grid[self.channel_idx["onion_supply"], y, x] = 1
                elif cell == "D":
                    # Dish supply
                    grid[self.channel_idx["dish_supply"], y, x] = 1
                elif cell == "S":
                    # Serving area
                    grid[self.channel_idx["serve_area"], y, x] = 1
                elif cell == "P":
                    # Pot
                    grid[self.channel_idx["pot"], y, x] = 1
        
        # Add player positions and orientations
        for i, player in enumerate(players):
            pos = player["position"]
            orientation = player["orientation"]
            
            # Skip if player position is outside our grid
            if pos[0] >= self.grid_height or pos[1] >= self.grid_width:
                continue
                
            # Set player position
            grid[self.channel_idx["player_pos"], pos[0], pos[1]] = 1
            
            # Encode orientation as a fraction of 2π
            # North = 0, East = 0.25, South = 0.5, West = 0.75
            orientation_val = 0
            if orientation == "NORTH":
                orientation_val = 0
            elif orientation == "EAST":
                orientation_val = 0.25
            elif orientation == "SOUTH":
                orientation_val = 0.5
            elif orientation == "WEST":
                orientation_val = 0.75
            
            grid[self.channel_idx["player_orientation"], pos[0], pos[1]] = orientation_val
        
        # Add objects (onions, dishes, soups)
        for obj_pos, obj in objects.items():
            pos = eval(obj_pos)  # Convert string position to tuple
            
            # Skip if object position is outside our grid
            if pos[0] >= self.grid_height or pos[1] >= self.grid_width:
                continue
                
            obj_name = obj["name"]
            
            if obj_name == "onion":
                grid[self.channel_idx["onion"], pos[0], pos[1]] = 1
            elif obj_name == "dish":
                grid[self.channel_idx["dish"], pos[0], pos[1]] = 1
            elif obj_name == "soup":
                grid[self.channel_idx["soup"], pos[0], pos[1]] = 1
                # Could also encode soup state (cooking time, ingredients) here if needed
        
        return grid
        
    def get_grid_dimensions(self):
        """Return the current grid dimensions."""
        return self.grid_height, self.grid_width 