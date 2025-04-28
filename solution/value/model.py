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
         -> Conv2d(31→32, k=2, s=1, p=0) -> ReLU       => (B, 32, 13, 5)
         -> Conv2d(32→64, k=2, s=1, p=0) -> ReLU       => (B, 64, 12, 4)
         -> Conv2d(64→64, k=2, s=1, p=0) -> ReLU       => (B, 64, 11, 3)
         -> Flatten                              => (B, 64*11*3)
      -> head:
         -> Linear(64*11*3 → CNN_FEATURE_SIZE) -> ReLU
         -> Linear(CNN_FEATURE_SIZE → n_act)
      Output wrapped as .q_value
    """
    def __init__(self, obs_shape, n_act):
        super().__init__()
        c, h, w = obs_shape
        
        # Create encoder layers
        encoder_layers = []
        in_c = c
        for out_c, k, s in CNN_FILTERS:
            pad = (k - 1) // 2
            encoder_layers.append(nn.Conv2d(in_c, out_c, k, s, pad))
            encoder_layers.append(nn.ReLU())
            # update spatial dims
            h = int(np.floor((h + 2 * pad - k) / s + 1))
            w = int(np.floor((w + 2 * pad - k) / s + 1))
            in_c = out_c
        
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate output feature size from encoder
        feature_size = in_c * h * w
        
        # Create head layers
        self.head = nn.Sequential(
            nn.Linear(feature_size, CNN_FEATURE_SIZE),
            nn.ReLU(),
            nn.Linear(CNN_FEATURE_SIZE, n_act)
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.head(features)
        return QValueOutput(output)