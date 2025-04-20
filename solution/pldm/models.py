class DynamicsNetwork(nn.Module):
    """
    Neural network for predicting the next state given the current state and action.
    Uses a convolutional architecture for processing grid-based state representations.
    """
    def __init__(
        self,
        num_channels,
        grid_height,
        grid_width,
        action_dim,
        hidden_size=128,
        num_layers=3
    ):
        super().__init__()
        
        # Store grid dimensions
        self.num_channels = num_channels
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.action_dim = action_dim
        
        # Convolutional layers for state processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Calculate flattened size after convolutions
        self.conv_output_size = 64 * grid_height * grid_width
        
        # Action embedding
        self.action_embedding = nn.Linear(action_dim, 64)
        
        # Fully connected layers for state-action processing
        fc_layers = []
        
        # Input to first layer is conv output + action embedding
        input_size = self.conv_output_size + 64
        
        # Add hidden layers
        for _ in range(num_layers - 1):
            fc_layers.append(nn.Linear(input_size, hidden_size))
            fc_layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer maps to next state representation
        fc_layers.append(nn.Linear(hidden_size, num_channels * grid_height * grid_width))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self, state, action):
        """
        Forward pass through the dynamics network.
        
        Args:
            state: Grid state representation [batch_size, num_channels, grid_height, grid_width]
            action: Action vector [batch_size, action_dim]
            
        Returns:
            Predicted next state in the same format as input state
        """
        batch_size = state.shape[0]
        
        # Process state through convolutional layers
        state_features = self.conv_layers(state)
        state_features = state_features.reshape(batch_size, -1)  # Flatten
        
        # Process action
        action_features = F.relu(self.action_embedding(action))
        
        # Combine state and action features
        combined = torch.cat([state_features, action_features], dim=1)
        
        # Process through fully connected layers
        output = self.fc_layers(combined)
        
        # Reshape output to match state format
        output = output.reshape(batch_size, self.num_channels, self.grid_height, self.grid_width)
        
        return output

class RewardNetwork(nn.Module):
    """
    Neural network for predicting the reward given the current state and action.
    Uses a convolutional architecture for processing grid-based state representations.
    """
    def __init__(
        self,
        num_channels,
        grid_height,
        grid_width,
        action_dim,
        hidden_size=128,
        num_layers=3
    ):
        super().__init__()
        
        # Store dimensions
        self.num_channels = num_channels
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.action_dim = action_dim
        
        # Convolutional layers for state processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Calculate flattened size after convolutions
        self.conv_output_size = 64 * grid_height * grid_width
        
        # Action embedding
        self.action_embedding = nn.Linear(action_dim, 64)
        
        # Fully connected layers
        fc_layers = []
        
        # Input to first layer is conv output + action embedding
        input_size = self.conv_output_size + 64
        
        # Add hidden layers
        for _ in range(num_layers - 1):
            fc_layers.append(nn.Linear(input_size, hidden_size))
            fc_layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer predicts a single reward value
        fc_layers.append(nn.Linear(hidden_size, 1))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self, state, action):
        """
        Forward pass through the reward network.
        
        Args:
            state: Grid state representation [batch_size, num_channels, grid_height, grid_width]
            action: Action vector [batch_size, action_dim]
            
        Returns:
            Predicted reward [batch_size, 1]
        """
        batch_size = state.shape[0]
        
        # Process state through convolutional layers
        state_features = self.conv_layers(state)
        state_features = state_features.reshape(batch_size, -1)  # Flatten
        
        # Process action
        action_features = F.relu(self.action_embedding(action))
        
        # Combine state and action features
        combined = torch.cat([state_features, action_features], dim=1)
        
        # Process through fully connected layers to get reward prediction
        reward = self.fc_layers(combined)
        
        return reward.squeeze(-1)  # Remove last dimension to get [batch_size] 