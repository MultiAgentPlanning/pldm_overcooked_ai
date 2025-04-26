# config.py

import d3rlpy

# ========================================================================
# Constants and Configuration
# ========================================================================

# Define the number of discrete actions per agent
NUM_ACTIONS_PER_AGENT = 6
TOTAL_JOINT_ACTIONS = NUM_ACTIONS_PER_AGENT * NUM_ACTIONS_PER_AGENT # 6 * 6 = 36

# --- d3rlpy Algorithm Configuration ---
SUPPORTED_ALGORITHMS = {
    "nfq": d3rlpy.algos.NFQConfig,
    "dqn": d3rlpy.algos.DQNConfig,
    "double_dqn": d3rlpy.algos.DoubleDQNConfig,
    "discrete_sac": d3rlpy.algos.DiscreteSACConfig,  # if available in your version
    "bcq": d3rlpy.algos.DiscreteBCQConfig,      # Batch-Constrained Q-Learning (advanced)
    "cql": d3rlpy.algos.DiscreteCQLConfig,
}

# --- CNN Encoder Configuration ---
# Use smaller 2x2 kernels to handle small input dimensions (W=6)
CNN_FILTERS = [(32, 2, 1), (64, 2, 1), (64, 2, 1)] # (out_channels, kernel_size=2, stride=1)
CNN_FEATURE_SIZE = 512 # Keep feature size, or adjust if needed based on new conv output size

