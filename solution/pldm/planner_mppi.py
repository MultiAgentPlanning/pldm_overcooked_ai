import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging # Import logging
import time # Import time

from .state_encoder import GridStateEncoder, VectorStateEncoder
from .dynamics_predictor import GridDynamicsPredictor, VectorDynamicsPredictor
from .reward_predictor import GridRewardPredictor, VectorRewardPredictor
from .utils import set_seeds

# Get logger for this module
logger = logging.getLogger(__name__)

class Planner:
    """
    Implements a sampling-based Model Predictive Control (MPC) planner
    using learned dynamics and reward models.
    """
    def __init__(self,
                 dynamics_model: torch.nn.Module,
                 reward_model: torch.nn.Module,
                 state_encoder, # Should be GridStateEncoder or VectorStateEncoder instance
                 num_actions: int = 6,
                 planning_horizon: int = 10,
                 num_samples: int = 100,
                 lambda_temp: float = 1.0,
                 device: Optional[torch.device] = None,
                 seed: Optional[int] = None):
        """
        Initialize the Planner.

        Args:
            dynamics_model: Trained dynamics prediction model.
            reward_model: Trained reward prediction model.
            state_encoder: Initialized state encoder instance.
            num_actions: Number of possible discrete actions per agent.
            planning_horizon: Number of steps to simulate into the future.
            num_samples: Number of action sequences to sample and evaluate.
            lambda_temp: control temperature for weighting trajectory rewards
            device: PyTorch device to run computations on.
            seed: Random seed for reproducible action sampling.
        """
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.state_encoder = state_encoder
        self.num_actions = num_actions
        self.horizon = planning_horizon
        self.num_samples = num_samples
        self.lambda_temp = lambda_temp
        
        # Store seed for later use
        self.seed = seed
        if seed is not None:
            logger.info(f"Setting planner seed to {seed}")
            set_seeds(seed)
        
        # Determine device or use model's device
        if device is not None:
            self.device = device
        else:
            try:
                # Try to infer device from model parameters
                self.device = next(self.dynamics_model.parameters()).device
            except Exception:
                logger.warning("Could not infer device from model. Defaulting to CPU.")
                self.device = torch.device('cpu')

        # Create random generator with the correct device after device is determined
        if seed is not None:
            try:
                # Try creating a generator with the specified device (newer PyTorch versions)
                self.rng = torch.Generator(device=self.device)
                self.rng.manual_seed(seed)
            except (TypeError, RuntimeError) as e:
                # Fallback for older PyTorch versions or unsupported devices
                if self.device.type != 'cpu':
                    logger.warning(f"Cannot create generator on {self.device} due to: {e}")
                    logger.warning("Falling back to CPU generator. Planning might need to use CPU tensors.")
                    # If the device is not CPU, we might need to move tensors to CPU during planning
                    self.device_mismatch = True
                self.rng = torch.Generator()
                self.rng.manual_seed(seed)
        else:
            self.rng = None
            self.device_mismatch = False

        # Ensure models are on the correct device and in eval mode
        self.dynamics_model.to(self.device).eval()
        self.reward_model.to(self.device).eval()

        logger.info(f"Planner initialized with horizon={self.horizon}, samples={self.num_samples} on device={self.device}")

    def _sample_action_sequences(self) -> torch.Tensor:
        """
        Sample random joint-action sequences.

        Returns:
            Tensor of shape [num_samples, horizon, 2] with action indices.
        """
        # Sample random actions for agent 1 and agent 2 independently -> sampling should be done from the Gaussian
        if self.rng is not None:
            # Check if we have a device mismatch (generator on CPU but tensors on another device)
            if hasattr(self, 'device_mismatch') and self.device_mismatch:
                # Sample on CPU first
                action_sequences = torch.randint(
                    low=0,
                    high=self.num_actions,
                    size=(self.num_samples, self.horizon, 2),
                    dtype=torch.long,
                    generator=self.rng  # This is a CPU generator
                )
                # Then move to the target device
                action_sequences = action_sequences.to(self.device)
            else:
                # Use deterministic generator if seed was provided (no device mismatch)
                action_sequences = torch.randint(
                    low=0,
                    high=self.num_actions,
                    size=(self.num_samples, self.horizon, 2),
                    dtype=torch.long,
                    device=self.device,
                    generator=self.rng
                )
        else:
            # Use default RNG otherwise
            action_sequences = torch.randint(
                low=0,
                high=self.num_actions,
                size=(self.num_samples, self.horizon, 2),
                dtype=torch.long,
                device=self.device
            )
        logger.debug(f"Sampled {action_sequences.shape[0]} action sequences of horizon {action_sequences.shape[1]}")
        return action_sequences

    
    # To modify as per encoder, reward_model, and dynamics_model used
    @torch.no_grad()
    def _simulate_trajectories(self, current_state_dict: Dict, action_sequences: torch.Tensor) -> torch.Tensor:
        """
        Simulate trajectories using the learned models and calculate cumulative rewards.

        Args:
            current_state_dict: The current state of the environment as a dictionary.
            action_sequences: Tensor of shape [num_samples, horizon, 2] containing action indices.

        Returns:
            Tensor of shape [num_samples] containing the total predicted reward for each trajectory.
        """
        batch_size = self.num_samples # Corresponds to num_samples trajectories
        total_rewards = torch.zeros(batch_size, device=self.device)
        
        try:
            # Encode the initial state
            current_state_encoded = self.state_encoder.encode(current_state_dict)
            current_state_tensor = torch.tensor(current_state_encoded, dtype=torch.float32, device=self.device)
        except Exception as e:
            logger.error(f"Error encoding initial state: {e}")
            raise # Re-raise exception as simulation cannot proceed

        # Repeat the initial state for each sampled trajectory
        # Shape: [num_samples, channels, height, width] or [num_samples, state_dim]
        if current_state_tensor.dim() == 3: # Grid state
             current_states_batch = current_state_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        elif current_state_tensor.dim() == 1: # Vector state
             current_states_batch = current_state_tensor.unsqueeze(0).repeat(batch_size, 1)
        else:
            logger.error(f"Unexpected initial state tensor dimension: {current_state_tensor.dim()}")
            raise ValueError("Initial state tensor has unexpected dimensions.")
        
        logger.debug(f"Starting trajectory simulation for {batch_size} samples over horizon {self.horizon}")

        # Simulate each step in the horizon
        for t in range(self.horizon):
            # Get actions for the current step for all samples
            # Shape: [num_samples, 2]
            current_actions = action_sequences[:, t, :]

            try:
                # Predict reward for the current state and action
                predicted_rewards = self.reward_model(current_states_batch, current_actions)
                if predicted_rewards.dim() > 1:
                     predicted_rewards = predicted_rewards.squeeze(-1) # Remove trailing dimension if exists
                total_rewards += predicted_rewards

                # Predict the next state for all samples
                predicted_next_states = self.dynamics_model(current_states_batch, current_actions)

                # Update current states for the next iteration
                current_states_batch = predicted_next_states
                
            except Exception as step_error:
                logger.error(f"Error during simulation step {t}: {step_error}")
                # Return partial rewards calculated so far, or handle as needed
                # For simplicity, we stop simulation here if an error occurs.
                # You might want to return NaNs or partial results depending on use case.
                return total_rewards # Return rewards accumulated up to the error point

            # Debug log for state shapes (optional)
            # logger.debug(f"  Step {t}: States shape {current_states_batch.shape}")

        logger.debug("Finished trajectory simulation.")
        return total_rewards

    def plan(self, current_state_dict: Dict) -> Tuple[np.ndarray, float]:
        """
        Plan the best joint action for the current state using MPC.

        Args:
            current_state_dict: The current state of the environment as a dictionary.

        Returns:
            Tuple containing:
                - The best joint action as a numpy array of shape [2,].
                - The predicted cumulative reward for the best trajectory (float).
        """
        logger.info(f"Planning started for state (details omitted)... Horizon={self.horizon}, Samples={self.num_samples}")
        start_time = time.time()
        
        # 1. Sample action sequences
        action_sequences = self._sample_action_sequences() # Shape: [num_samples, horizon, 2]

        # 2. Simulate trajectories and get cumulative rewards
        try:
            cumulative_rewards = self._simulate_trajectories(current_state_dict, action_sequences) # Shape: [num_samples]
        except Exception as sim_err:
            logger.error(f"Trajectory simulation failed: {sim_err}. Cannot plan.")
            # Return a default action (e.g., STAY) and zero reward if planning fails
            return np.array([0, 0], dtype=np.int64), 0.0 


        # 3. Compute costs (negative rewards)
        costs = -cumulative_rewards  # [num_samples]

        # 4. Softmax weighting
        weights = torch.softmax(-costs / self.lambda_temp, dim=0)  # [num_samples]

        # 5. Weighted action voting to find best action
        # Collect all first actions
        first_actions = action_sequences[:, 0, :]  # [num_samples, 2]

        final_action = torch.zeros((2,), device=self.device)

        for agent_idx in range(2):
            agent_actions = first_actions[:, agent_idx]  # [num_samples]
            # Weighted histogram over discrete actions
            weighted_counts = torch.zeros((self.num_actions,), device=self.device)
            for a in range(self.num_actions):
                mask = (agent_actions == a)
                weighted_counts[a] = (weights * mask.float()).sum()

            # Pick action with highest weighted count
            best_action = torch.argmax(weighted_counts)
            final_action[agent_idx] = best_action

        
        # Also return best expected reward for monitoring
        best_reward_predicted = cumulative_rewards.max().item()

        end_time = time.time()
        logger.info(f"Planning finished in {end_time - start_time:.3f}s. Best predicted reward: {best_reward_predicted:.4f}")
        
        return final_action.long().cpu().numpy(), best_reward_predicted # Return action and predicted reward 