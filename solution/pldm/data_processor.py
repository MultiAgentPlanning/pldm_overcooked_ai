import os
import csv
import json
import torch
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Union, Optional
import logging # Import logging
from tqdm import tqdm # Import tqdm

from .utils import parse_state, parse_joint_action, parse_layout, get_action_index
from .state_encoder import GridStateEncoder, VectorStateEncoder

# Get logger for this module
logger = logging.getLogger(__name__)

# Custom collate function to handle tensors of different sizes
def custom_collate_fn(batch, return_terminal: bool = False):
    """
    Custom collate function that handles tensors of different sizes.
    For grid states, it pads all tensors to the maximum size in the batch.
    
    Args:
        batch: List of (state, action, next_state, reward) tuples
        
    Returns:
        Tuple of batched tensors
    """
    # Separate the batch into components
    if return_terminal:
        states, actions, next_states, rewards, terms = zip(*batch)
    else:
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
    
    if return_terminal:
        batched_terms = torch.stack(terms)
        return batched_states, batched_actions, batched_next_states, batched_rewards, batched_terms
    
    return batched_states, batched_actions, batched_next_states, batched_rewards


def reshape_tensor(tensor, target_size):
    """
    Reshape a single tensor to the target 2D size using padding.
    
    Args:
        tensor: Input tensor of shape (channels, height, width)
        target_size: Tuple of (target_height, target_width)
        
    Returns:
        Reshaped tensor of shape (channels, target_height, target_width)
    """
    # Extract target dimensions
    target_height, target_width = target_size
    
    # Get current dimensions
    channels, height, width = tensor.size()

    assert target_height >= height and target_width >= width, \
        f"Target size {target_size} must be greater than or equal to current size {(channels, height, width)}"
    
    # Calculate padding needed
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)
    
    # Apply padding if needed
    if pad_height > 0 or pad_width > 0:
        # Padding format: (left, right, top, bottom, front, back)
        padding = (0, pad_width, 0, pad_height, 0, 0)
        reshaped_tensor = torch.nn.functional.pad(tensor, padding)
    else:
        reshaped_tensor = tensor
    
    return reshaped_tensor


class OvercookedDataset(Dataset):
    """
    Dataset for Overcooked transitions (state, action, next_state, reward).
    Stores episode boundary information for sequence retrieval.
    """
    def __init__(self, 
                 data_path: str,
                 state_encoder_type: str = 'grid',
                 max_samples: Optional[int] = None,
                 return_terminal: bool = False):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to CSV dataset
            state_encoder_type: Type of state encoder ('grid' or 'vector')
            max_samples: Maximum number of samples to load (optional, for testing)
        """
        self.data_path = data_path
        self.state_encoder_type = state_encoder_type
        logger.info(f"Initializing OvercookedDataset: path={data_path}, type={state_encoder_type}, max_samples={max_samples}")
        
        # Initialize state encoder
        if state_encoder_type == 'grid':
            self.state_encoder = GridStateEncoder()
        elif state_encoder_type == 'vector':
            self.state_encoder = VectorStateEncoder()
        else:
            logger.error(f"Unknown state encoder type: {state_encoder_type}")
            raise ValueError(f"Unknown state encoder type: {state_encoder_type}")
        
        # Load and process data
        self.transitions = [] # List to store (state_enc, action_indices, next_state_enc, reward)
        self.episode_info = [] # List to store (start_idx, end_idx, trial_id)
        self._load_data(max_samples)
        logger.info(f"Loaded {len(self.transitions)} transitions across {len(self.episode_info)} episodes.")
        
        self.return_terminal = return_terminal
        if self.return_terminal:
            self.terminals = np.zeros(len(self.transitions), dtype=np.float32)
            for start_idx, end_idx, _ in self.episode_info:
                self.terminals[end_idx] = 1.0
    
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
             logger.info(f"Found simple cache {simple_cache_path}. Removing to regenerate with episode info.")
             try:
                 os.remove(simple_cache_path)
             except OSError as e:
                 logger.error(f"Error removing cache file: {e}")

        # --- Start processing --- 
        logger.info(f"Processing data from {self.data_path} to include episode info.")
        current_transition_index = 0
        
        try:
            df = pd.read_csv(self.data_path)
            # Count trials for tqdm progress bar
            num_trials = df['trial_id'].nunique()
            trials = df.groupby('trial_id')
            n_trials_processed = 0
            
            # Use tqdm to show progress over trials
            trial_iterator = tqdm(trials, total=num_trials, desc="Processing Trials", unit="trial")

            for trial_id, trial_df in trial_iterator:
                # Sort by timestep
                def extract_timestep(state_json):
                    try:
                        return json.loads(state_json).get("timestep", 0)
                    except json.JSONDecodeError:
                         logger.warning(f"Malformed JSON found in trial {trial_id}. Returning timestep 0.")
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
                        
                        # Ensure timestep is in the state_dict
                        if 'extracted_timestep' in curr_row:
                            curr_state_dict['timestep'] = curr_row['extracted_timestep']
                        elif 'cur_gameloop' in curr_row:
                            curr_state_dict['timestep'] = curr_row['cur_gameloop']
                        else:
                            # If no explicit timestep, use the row index within the trial
                            curr_state_dict['timestep'] = i
                        
                        # Ensure timestep is in the next_state_dict
                        if 'extracted_timestep' in next_row:
                            next_state_dict['timestep'] = next_row['extracted_timestep']
                        elif 'cur_gameloop' in next_row:
                            next_state_dict['timestep'] = next_row['cur_gameloop']
                        else:
                            # If no explicit timestep, use the row index within the trial
                            next_state_dict['timestep'] = i + 1
                        
                        # Parse layout data if available
                        if 'layout' in curr_row and curr_row['layout']:
                            try:
                                layout = parse_layout(curr_row['layout'])
                                curr_state_dict['layout'] = layout
                            except Exception as layout_error:
                                logger.warning(f"Error parsing layout in trial {trial_id}, row {i}: {layout_error}")
                        
                        # Parse layout data for next state if available
                        if 'layout' in next_row and next_row['layout']:
                            try:
                                layout = parse_layout(next_row['layout'])
                                next_state_dict['layout'] = layout
                            except Exception as layout_error:
                                logger.warning(f"Error parsing layout in trial {trial_id}, row {i+1}: {layout_error}")
                        
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
                        logger.warning(f"Skipping row {i} in trial {trial_id} due to error: {row_error}")
                        continue # Skip problematic rows

                    # Break if we have enough total samples
                    if max_samples and current_transition_index >= max_samples:
                        break
                
                # Record episode info if any transitions were added
                if num_transitions_in_episode > 0:
                    episode_end_index = current_transition_index - 1
                    self.episode_info.append((episode_start_index, episode_end_index, trial_id))
                
                n_trials_processed += 1
                # Update tqdm description
                trial_iterator.set_description(f"Processed {n_trials_processed}/{num_trials} trials")
                trial_iterator.set_postfix(transitions=f"{current_transition_index}")
                
                # Break outer loop if max_samples reached
                if max_samples and current_transition_index >= max_samples:
                    logger.info(f"Reached max_samples ({max_samples}). Stopping data processing.")
                    break
            
            trial_iterator.close()
            # We are not re-implementing caching here for simplicity.
            # A robust implementation would cache self.transitions and self.episode_info.
            if not self.transitions:
                 logger.warning("No transitions were loaded after processing.")

        except FileNotFoundError:
             logger.error(f"Error: Data file not found at {self.data_path}")
        except Exception as e:
            import traceback
            logger.error(f"Error processing data: {e}")
            logger.error(traceback.format_exc())
            
    def get_episode_info(self, transition_index: int) -> Optional[Tuple[int, int, str]]:
        """Find the episode info (start_idx, end_idx, trial_id) for a given transition index."""
        for start_idx, end_idx, trial_id in self.episode_info:
            if start_idx <= transition_index <= end_idx:
                return start_idx, end_idx, trial_id
        logger.warning(f"Could not find episode info for transition index {transition_index}")
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
        
        if self.return_terminal:
            term_tensor = torch.tensor(self.terminals[idx], dtype=torch.float32)
            return state_tensor, action_tensor, next_state_tensor, reward_tensor, term_tensor
        
        return state_tensor, action_tensor, next_state_tensor, reward_tensor


def get_overcooked_dataloaders(data_path: str, 
                               state_encoder_type: str = 'grid',
                               batch_size: int = 64,
                               val_ratio: float = 0.1,
                               test_ratio: float = 0.1,
                               max_samples: Optional[int] = None,
                               return_terminal: bool = False,
                               num_workers: int = 4,
                               seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for Overcooked data.
    
    Args:
        data_path: Path to CSV dataset
        state_encoder_type: Type of state encoder ('grid' or 'vector') 
        batch_size: Batch size for dataloaders
        val_ratio: Ratio of data to use for validation (default: 0.1)
        test_ratio: Ratio of data to use for testing (default: 0.1)
        max_samples: Maximum number of samples to load (optional, for testing)
        num_workers: Number of workers for dataloader
        seed: Random seed for reproducible dataset splitting
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = OvercookedDataset(data_path, state_encoder_type, max_samples, return_terminal)
    
    # Handle empty dataset case
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! No transitions were loaded from the data file.")
    
    # Calculate sizes for each split
    test_size = max(1, int(test_ratio * len(dataset)))
    val_size = max(1, int(val_ratio * len(dataset)))
    train_size = len(dataset) - val_size - test_size
    
    # Use the generator from torch for reproducible splits if seed is provided
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        logger.info(f"Using seed {seed} for dataset splitting")
    
    # Split into train, validation, and test datasets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )
    
    # Log split information
    logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test samples")
    
    # Create dataloaders with custom collate function
    collate = partial(custom_collate_fn, return_terminal=return_terminal)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate
    )
    
    return train_loader, val_loader, test_loader