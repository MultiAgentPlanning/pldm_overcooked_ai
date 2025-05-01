import torch.nn as nn
import numpy as np
from solution.value.config import CNN_FILTERS, CNN_FEATURE_SIZE

class QValueOutput:
    def __init__(self, q_value):
        self.q_value = q_value

class QValueNet(nn.Module):
    """
    Q-network architecture (example for obs_shape=(31,14,6)):
      Input: (B, C=31, H=14, W=6)
      -> encoder:
         -> Linear(32) -> ReLU       => (B, 32)
      -> head:
         -> Linear(32 → 32) -> ReLU
         -> Linear(32 → n_act)
      Output wrapped as .q_value
    """
    def __init__(self, obs_shape, action_size):
        super(QValueNet, self).__init__()
        self.action_size = action_size
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(obs_shape), 32),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.encoder(x)
        q_value = self.head(x)
        return QValueOutput(q_value)