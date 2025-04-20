import os
import csv
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Union, Optional

from .utils import parse_state, parse_joint_action, parse_layout, get_action_index
from .state_encoder import GridStateEncoder, VectorStateEncoder


# Custom collate function to handle tensors of different sizes
def custom_collate_fn(batch):
    """
    Custom collate function that handles tensors of different sizes.
    For grid states, it pads all tensors to the maximum size in the batch.
    
    Args:
        batch: List of (state, action, next_state, reward) tuples
        
    Returns:
        Tuple of batched tensors
    """
    # Separate the batch into components
    states, actions, next_states, rewards = zip(*batch)
    
    # Check if we have grid or vector states
    if states[0].dim() == 3:  # Grid states (channels, height, width)
        # Find maximum dimensions
        max_channels = max(s.size(0) for s in states)
        max_height = max(s.size(1) for s in states)
        max_width = max(s.size(2) for s in states)
        
        # Pad and stack states
        padded_states = []
        for s in states:
            channels, height, width = s.size()
            # Create padding
            pad_channels = max_channels - channels
            pad_height = max_height - height
            pad_width = max_width - width
            
            # Apply padding if needed
            if pad_channels > 0 or pad_height > 0 or pad_width > 0:
                padding = (0, pad_width, 0, pad_height, 0, pad_channels)
                padded_s = torch.nn.functional.pad(s, padding)
                padded_states.append(padded_s)
            else:
                padded_states.append(s)
        
        batched_states = torch.stack(padded_states)
        
        # Same for next_states
        padded_next_states = []
        for s in next_states:
            channels, height, width = s.size()
            # Create padding
            pad_channels = max_channels - channels
            pad_height = max_height - height
            pad_width = max_width - width
            
            # Apply padding if needed
            if pad_channels > 0 or pad_height > 0 or pad_width > 0:
                padding = (0, pad_width, 0, pad_height, 0, pad_channels)
                padded_s = torch.nn.functional.pad(s, padding)
                padded_next_states.append(padded_s)
            else:
                padded_next_states.append(s)
        
        batched_next_states = torch.stack(padded_next_states)
    else:  # Vector states
        # Just stack vectors (they should be consistent anyway)
        batched_states = torch.stack(states)
        batched_next_states = torch.stack(next_states)
    
    # Batch actions and rewards (should be consistent sizes)
    batched_actions = torch.stack(actions)
    batched_rewards = torch.stack(rewards)
    
    return batched_states, batched_actions, batched_next_states, batched_rewards


class OvercookedDataset(Dataset):
    """
    Dataset for Overcooked transitions (state, action, next_state, reward).
    Stores episode boundary information for sequence retrieval.
    """
    def __init__(self, 
                 data_path: str,
                 state_encoder_type: str = 'grid',
                 max_samples: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to CSV dataset
            state_encoder_type: Type of state encoder ('grid' or 'vector')
            max_samples: Maximum number of samples to load (optional, for testing)
        """
        self.data_path = data_path
        self.state_encoder_type = state_encoder_type
        
        # Initialize state encoder
        if state_encoder_type == 'grid':
            self.state_encoder = GridStateEncoder()
        elif state_encoder_type == 'vector':
            self.state_encoder = VectorStateEncoder()
        else:
            raise ValueError(f"Unknown state encoder type: {state_encoder_type}")
        
        # Load and process data
        self.transitions = [] # List to store (state_enc, action_indices, next_state_enc, reward)
        self.episode_info = [] # List to store (start_idx, end_idx, trial_id)
        self._load_data(max_samples)
        print(f"Loaded {len(self.transitions)} transitions across {len(self.episode_info)} episodes.")
    
    def _load_data(self, max_samples: Optional[int] = None):
        """
        Load data from CSV and create state transition pairs.
        Populates self.transitions and self.episode_info.
        
        Args:
            max_samples: Maximum number of samples to load
        """
        # Check if a *different* kind of cache exists (that includes episode info)
        # For simplicity, we'll just regenerate if the simple cache exists or no cache exists.
        # A more robust caching mechanism could be added later.
        base_cache_path = f"{os.path.splitext(self.data_path)[0]}_{self.state_encoder_type}"
        simple_cache_path = f"{base_cache_path}_cache.npz"
        # episode_cache_path = f"{base_cache_path}_episode_cache.npz" # Example for future

        # If simple cache exists, maybe delete it to force regeneration with episode info
        if os.path.exists(simple_cache_path):
             print(f"Found simple cache {simple_cache_path}. Removing to regenerate with episode info.")
             try:
                 os.remove(simple_cache_path)
             except OSError as e:
                 print(f"Error removing cache file: {e}")

        # --- Start processing --- 
        print(f"Processing data from {self.data_path} to include episode info.")
        current_transition_index = 0
        
        try:
            df = pd.read_csv(self.data_path)
            trials = df.groupby('trial_id')
            n_trials_processed = 0
            
            for trial_id, trial_df in trials:
                # Sort by timestep
                def extract_timestep(state_json):
                    try:
                        return json.loads(state_json).get("timestep", 0)
                    except json.JSONDecodeError:
                        return 0 # Handle malformed JSON
                
                trial_df = trial_df.copy()
                trial_df['extracted_timestep'] = trial_df['state'].apply(extract_timestep)
                trial_df = trial_df.sort_values('extracted_timestep')
                
                episode_start_index = current_transition_index
                num_transitions_in_episode = 0

                for i in range(len(trial_df) - 1):
                    curr_row = trial_df.iloc[i]
                    next_row = trial_df.iloc[i+1]
                    
                    try:
                        curr_state_dict = parse_state(curr_row['state'])
                        joint_action = parse_joint_action(curr_row['joint_action'])
                        next_state_dict = parse_state(next_row['state'])
                        reward = float(curr_row['reward'])
                        
                        action_indices = [
                            get_action_index(joint_action[0]),
                            get_action_index(joint_action[1])
                        ]
                        
                        curr_state_enc = self.state_encoder.encode(curr_state_dict)
                        next_state_enc = self.state_encoder.encode(next_state_dict)
                        
                        # Store transition data (including the reward)
                        self.transitions.append((
                            curr_state_enc,
                            np.array(action_indices),
                            next_state_enc,
                            reward # Store reward directly
                        ))
                        current_transition_index += 1
                        num_transitions_in_episode += 1
                        
                    except Exception as row_error:
                        print(f"Skipping row {i} in trial {trial_id} due to error: {row_error}")
                        continue # Skip problematic rows

                    # Break if we have enough total samples
                    if max_samples and current_transition_index >= max_samples:
                        break
                
                # Record episode info if any transitions were added
                if num_transitions_in_episode > 0:
                    episode_end_index = current_transition_index - 1
                    self.episode_info.append((episode_start_index, episode_end_index, trial_id))
                
                n_trials_processed += 1
                if n_trials_processed % 10 == 0:
                    print(f"Processed {n_trials_processed} trials, {current_transition_index} transitions")
                
                # Break outer loop if max_samples reached
                if max_samples and current_transition_index >= max_samples:
                    break
            
            # We are not re-implementing caching here for simplicity.
            # A robust implementation would cache self.transitions and self.episode_info.
            if not self.transitions:
                 print("Warning: No transitions were loaded after processing.")

        except FileNotFoundError:
             print(f"Error: Data file not found at {self.data_path}")
        except Exception as e:
            import traceback
            print(f"Error processing data: {e}")
            print(traceback.format_exc())
            
    def get_episode_info(self, transition_index: int) -> Optional[Tuple[int, int, str]]:
        """Find the episode info (start_idx, end_idx, trial_id) for a given transition index."""
        for start_idx, end_idx, trial_id in self.episode_info:
            if start_idx <= transition_index <= end_idx:
                return start_idx, end_idx, trial_id
        return None # Index out of bounds or not found

    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        # Returns tensors for DataLoader, reward is included here
        state_enc, action_indices, next_state_enc, reward = self.transitions[idx]
        
        state_tensor = torch.tensor(state_enc, dtype=torch.float32)
        action_tensor = torch.tensor(action_indices, dtype=torch.long)
        next_state_tensor = torch.tensor(next_state_enc, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32) 
        
        return state_tensor, action_tensor, next_state_tensor, reward_tensor


def get_overcooked_dataloaders(data_path: str, 
                               state_encoder_type: str = 'grid',
                               batch_size: int = 64,
                               val_ratio: float = 0.1,
                               max_samples: Optional[int] = None,
                               num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for Overcooked data.
    
    Args:
        data_path: Path to CSV dataset
        state_encoder_type: Type of state encoder ('grid' or 'vector') 
        batch_size: Batch size for dataloaders
        val_ratio: Ratio of data to use for validation
        max_samples: Maximum number of samples to load (optional, for testing)
        num_workers: Number of workers for dataloader
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = OvercookedDataset(data_path, state_encoder_type, max_samples)
    
    # Handle empty dataset case
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! No transitions were loaded from the data file.")
    
    # Split into train and validation
    val_size = max(1, int(val_ratio * len(dataset)))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader 