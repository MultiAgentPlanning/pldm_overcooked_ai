import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .state_encoder import GridStateEncoder, VectorStateEncoder
from .dynamics_predictor import GridDynamicsPredictor, VectorDynamicsPredictor
from .reward_predictor import GridRewardPredictor, VectorRewardPredictor

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
                 device: Optional[torch.device] = None):
        """
        Initialize the Planner.

        Args:
            dynamics_model: Trained dynamics prediction model.
            reward_model: Trained reward prediction model.
            state_encoder: Initialized state encoder instance.
            num_actions: Number of possible discrete actions per agent.
            planning_horizon: Number of steps to simulate into the future.
            num_samples: Number of action sequences to sample and evaluate.
            device: PyTorch device to run computations on.
        """
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.state_encoder = state_encoder
        self.num_actions = num_actions
        self.horizon = planning_horizon
        self.num_samples = num_samples

        # Determine device or use model's device
        if device is not None:
            self.device = device
        else:
            try:
                # Try to infer device from model parameters
                self.device = next(self.dynamics_model.parameters()).device
            except Exception:
                print("Warning: Could not infer device from model. Defaulting to CPU.")
                self.device = torch.device('cpu')

        # Ensure models are on the correct device and in eval mode
        self.dynamics_model.to(self.device).eval()
        self.reward_model.to(self.device).eval()

        print(f"Planner initialized with horizon={self.horizon}, samples={self.num_samples} on device={self.device}")

    def _sample_action_sequences(self) -> torch.Tensor:
        """
        Sample random joint-action sequences.

        Returns:
            Tensor of shape [num_samples, horizon, 2] with action indices.
        """
        # Sample random actions for agent 1 and agent 2 independently
        action_sequences = torch.randint(
            low=0,
            high=self.num_actions,
            size=(self.num_samples, self.horizon, 2),
            dtype=torch.long,
            device=self.device
        )
        return action_sequences

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

        # Encode the initial state
        current_state_encoded = self.state_encoder.encode(current_state_dict)
        current_state_tensor = torch.tensor(current_state_encoded, dtype=torch.float32, device=self.device)

        # Repeat the initial state for each sampled trajectory
        # Shape: [num_samples, channels, height, width] or [num_samples, state_dim]
        current_states_batch = current_state_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1) # For grid
        if current_states_batch.dim() == 2: # Adjust for vector states
             current_states_batch = current_state_tensor.unsqueeze(0).repeat(batch_size, 1)


        # Simulate each step in the horizon
        for t in range(self.horizon):
            # Get actions for the current step for all samples
            # Shape: [num_samples, 2]
            current_actions = action_sequences[:, t, :]

            # Predict reward for the current state and action
            # The reward model might need reshaping for its output
            predicted_rewards = self.reward_model(current_states_batch, current_actions)
            # Ensure reward is scalar per sample, shape [num_samples]
            if predicted_rewards.dim() > 1:
                 predicted_rewards = predicted_rewards.squeeze(-1) # Remove trailing dimension if exists

            total_rewards += predicted_rewards

            # Predict the next state for all samples
            # Shape: [num_samples, channels, height, width] or [num_samples, state_dim]
            predicted_next_states = self.dynamics_model(current_states_batch, current_actions)

            # Update current states for the next iteration
            current_states_batch = predicted_next_states

            # Handle potential dimension changes if models are dynamic (though less likely in eval)
            # Re-check shapes if necessary, but usually fixed after first pass in eval mode

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
        # 1. Sample action sequences
        action_sequences = self._sample_action_sequences() # Shape: [num_samples, horizon, 2]

        # 2. Simulate trajectories and get cumulative rewards
        cumulative_rewards = self._simulate_trajectories(current_state_dict, action_sequences) # Shape: [num_samples]

        # 3. Find the best trajectory
        best_trajectory_index = torch.argmax(cumulative_rewards).item()
        best_reward_predicted = cumulative_rewards[best_trajectory_index].item()

        # 4. Select the first action from the best trajectory
        best_first_action = action_sequences[best_trajectory_index, 0, :] # Shape: [2]

        return best_first_action.cpu().numpy(), best_reward_predicted # Return action and predicted reward 