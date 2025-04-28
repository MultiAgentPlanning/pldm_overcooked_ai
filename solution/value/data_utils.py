# data_utils.py

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Any, Optional
import ast
import os 

from solution.pldm.state_encoder import GridStateEncoder
from solution.pldm.utils import parse_state, parse_joint_action, get_action_index
from solution.value.config import NUM_ACTIONS_PER_AGENT

# Action mapping constants (N, S, E, W, Stay, Interact)
ACTION_MAP = {
    (-1, 0): 0, # Up (North)
    (1, 0): 1,  # Down (South)
    (0, 1): 3,  # Right (East)
    (0, -1): 2, # Left (West)
    (0, 0): 4,  # Stay
    "interact": 5,
}
ACTION_NAMES = ["North", "South", "West", "East", "Stay", "Interact"]

def map_raw_action_to_index(raw_action: Any) -> Optional[int]:
    """Maps a raw action representation to a discrete action index (0-5)."""
    if isinstance(raw_action, str):
        if raw_action.lower() == "interact":
            return ACTION_MAP["interact"]
        else:
            return None
    elif isinstance(raw_action, (list, tuple)) and len(raw_action) == 2 and all(isinstance(i, int) for i in raw_action):
        action_vec = tuple(raw_action)
        return ACTION_MAP.get(action_vec)
    else:
        return None

def find_max_dimensions(df: pd.DataFrame) -> Tuple[int, int]:
    """Finds max grid dimensions needed for padding."""
    max_h, max_w = 0, 0
    print("Pass 1: Finding maximum grid dimensions for padding...")
    for state_json_str in tqdm(df['state'], desc="Scanning States"):
        try:
            state_dict = parse_state(state_json_str) # Use user's parser
            for player in state_dict.get("players", []):
                pos = player.get("position")
                if pos and isinstance(pos, (list, tuple)) and len(pos) == 2:
                    max_h = max(max_h, pos[0] + 1)
                    max_w = max(max_w, pos[1] + 1)
            for obj in state_dict.get("objects", []):
                 pos = obj.get("position")
                 if pos and isinstance(pos, (list, tuple)) and len(pos) == 2:
                    max_h = max(max_h, pos[0] + 1)
                    max_w = max(max_w, pos[1] + 1)
            layout_grid = state_dict.get("layout")
            if layout_grid and isinstance(layout_grid, list) and layout_grid:
                max_h = max(max_h, len(layout_grid))
                if isinstance(layout_grid[0], list):
                     max_w = max(max_w, len(layout_grid[0]))
        except Exception:
             pass # Ignore errors during dimension scan
    max_h += 1
    max_w += 1
    print(f"Determined global max dimensions for padding: H={max_h}, W={max_w}")
    return max_h, max_w


def pad_grid(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pads a grid (C, H, W) to the target height and width."""
    if grid.ndim != 3:
        raise ValueError(f"Invalid grid dimension: expected 3 (C, H, W), got {grid.ndim}")
    c, h, w = grid.shape
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h < 0 or pad_w < 0:
        clipped_grid = grid[:, :min(h, target_h), :min(w, target_w)]
        h, w = clipped_grid.shape[1], clipped_grid.shape[2]
        pad_h = target_h - h
        pad_w = target_w - w
        grid_to_pad = clipped_grid
        if pad_h < 0 or pad_w < 0:
             raise RuntimeError("Internal error: Padding calculated incorrectly after clipping.")
    else:
        grid_to_pad = grid
    if pad_h == 0 and pad_w == 0:
        return grid_to_pad
    padding = ((0, 0), (0, max(0, pad_h)), (0, max(0, pad_w)))
    return np.pad(grid_to_pad, padding, mode='constant', constant_values=0)


# --- Main Data Processing Function (with Caching) ---

def process_data(df: pd.DataFrame, H_global_max: int, W_global_max: int, cache_dir: str = "./data/") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes the DataFrame to extract transitions and format them for d3rlpy.
    Uses imported functions parse_state, parse_joint_action, get_action_index.
    Groups by trial_id to correctly determine terminal states.
    Uses the imported GridStateEncoder and pads states globally.
    Implements caching to save/load processed data.

    Args:
        df: Input Pandas DataFrame.
        H_global_max: Target height for padding state grids.
        W_global_max: Target width for padding state grids.
        cache_dir: Directory to store cached processed data.

    Returns:
        A tuple containing NumPy arrays:
        (observations, actions, rewards, next_observations, terminals)

    Raises:
        RuntimeError: If processed array lengths don't match.
    """
    # --- Caching Logic ---
    # Create a unique cache filename based on data characteristics (optional)
    # For simplicity, just use a fixed name related to the dimensions
    cache_filename = f"overcooked_processed_H{H_global_max}_W{W_global_max}.npz"
    cache_path = os.path.join(cache_dir, cache_filename)

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Loading processed data from cache: {cache_path}")
        try:
            cached_data = np.load(cache_path)
            # Verify all expected keys are present
            required_keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals']
            if all(key in cached_data for key in required_keys):
                print("Cache loaded successfully.")
                return (cached_data['observations'], cached_data['actions'],
                        cached_data['rewards'], cached_data['next_observations'],
                        cached_data['terminals'])
            else:
                print("Warning: Cache file is missing required keys. Reprocessing data.")
        except Exception as e:
            print(f"Warning: Failed to load cache file ({e}). Reprocessing data.")
    else:
        print("No cache found. Processing data from scratch...")

    # --- Data Processing Logic (if cache miss or invalid) ---
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    processed_count = 0
    skipped_count = 0
    logged_action_format_issue = False

    encoder = GridStateEncoder()
    num_channels_expected = encoder.num_channels

    print("Pass 2: Processing trajectories and encoding data using provided utils...")
    grouped = df.groupby('trial_id')
    num_trajectories = len(grouped)

    with tqdm(total=num_trajectories, desc="Processing Trajectories") as pbar_traj:
        for trial_id, group in grouped:
            if 'extracted_timestep' in group.columns:
                 group = group.sort_values(by='extracted_timestep').reset_index(drop=True)
            else:
                 group = group.sort_values(by='cur_gameloop').reset_index(drop=True)
            group_len = len(group)

            if group_len < 2:
                skipped_count += group_len
                pbar_traj.update(1)
                continue

            for i in range(group_len - 1):
                current_row = group.iloc[i]
                next_row = group.iloc[i+1]

                try:
                    # State Parsing and Encoding
                    curr_state_dict = parse_state(current_row['state'])
                    state_grid = encoder.encode(curr_state_dict)
                    if state_grid.shape[0] != num_channels_expected:
                         raise ValueError(f"Encoder channel mismatch: {state_grid.shape[0]} vs {num_channels_expected}")
                    padded_state = pad_grid(state_grid, H_global_max, W_global_max)
                    observations.append(padded_state.astype(np.float32))

                    # Next State Parsing and Encoding
                    next_state_dict = parse_state(next_row['state'])
                    next_state_grid = encoder.encode(next_state_dict)
                    if next_state_grid.shape[0] != num_channels_expected:
                         raise ValueError(f"Encoder channel mismatch (next): {next_state_grid.shape[0]} vs {num_channels_expected}")
                    padded_next_state = pad_grid(next_state_grid, H_global_max, W_global_max)
                    next_observations.append(padded_next_state.astype(np.float32))

                    # Action Parsing and Mapping
                    parsed_action_pair = parse_joint_action(current_row['joint_action'])
                    if not isinstance(parsed_action_pair, (list, tuple)) or len(parsed_action_pair) != 2:
                         raise ValueError(f"parse_joint_action invalid output: {parsed_action_pair}")

                    action_idx_0 = get_action_index(parsed_action_pair[0])
                    action_idx_1 = get_action_index(parsed_action_pair[1])

                    if not (0 <= action_idx_0 < NUM_ACTIONS_PER_AGENT and 0 <= action_idx_1 < NUM_ACTIONS_PER_AGENT):
                        raise ValueError(f"get_action_index invalid index: ({action_idx_0}, {action_idx_1})")

                    # Combine into Joint Action Index for d3rlpy
                    joint_action_idx = action_idx_0 * NUM_ACTIONS_PER_AGENT + action_idx_1
                    actions.append(joint_action_idx)

                    # Reward
                    rewards.append(float(current_row['reward']))

                    # Terminal
                    is_terminal = (i == group_len - 2)
                    terminals.append(is_terminal)
                    processed_count += 1

                except Exception as e:
                    if skipped_count % 500 == 0:
                         error_context = f"Trial: {trial_id}, Step: {i}, ActionStr: {current_row.get('joint_action', 'N/A')}"
                         print(f"\nWarning: Skipping transition due to error ({error_context}): {type(e).__name__} - {e}")
                    skipped_count += 1

            pbar_traj.update(1)

    print(f"Finished processing. Processed transitions: {processed_count}, Skipped rows/transitions: {skipped_count}")

    if not observations:
         print("Error: No valid transitions were successfully processed.")
         obs_shape = (0, num_channels_expected, H_global_max, W_global_max)
         return (np.empty(obs_shape, dtype=np.float32), np.empty((0,), dtype=np.int32),
                 np.empty((0,), dtype=np.float32), np.empty(obs_shape, dtype=np.float32),
                 np.empty((0,), dtype=np.float32))

    print("Converting processed data to NumPy arrays...")
    np_observations = np.array(observations, dtype=np.float32)
    np_actions = np.array(actions, dtype=np.int32)
    np_rewards = np.array(rewards, dtype=np.float32)
    np_next_observations = np.array(next_observations, dtype=np.float32)
    np_terminals = np.array(terminals, dtype=np.float32)

    print(f"Final dataset shapes:")
    print(f"  Observations: {np_observations.shape}")
    print(f"  Actions: {np_actions.shape}")
    print(f"  Rewards: {np_rewards.shape}")
    print(f"  Next Observations: {np_next_observations.shape}") # Still needed for saving
    print(f"  Terminals: {np_terminals.shape}")

    n_transitions = len(np_observations)
    if not (len(np_actions) == n_transitions and len(np_rewards) == n_transitions and
            len(np_next_observations) == n_transitions and len(np_terminals) == n_transitions):
        raise RuntimeError(f"Mismatch in processed array lengths! Check processing logic.")

    # --- Save processed data to cache ---
    try:
        print(f"Saving processed data to cache: {cache_path}")
        np.savez_compressed(cache_path, # Use compressed format
                 observations=np_observations,
                 actions=np_actions,
                 rewards=np_rewards,
                 next_observations=np_next_observations, # Save next_obs too
                 terminals=np_terminals)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Warning: Failed to save processed data to cache ({cache_path}): {e}")

    return np_observations, np_actions, np_rewards, np_next_observations, np_terminals
