import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticConfig:
    """Configuration class for Critic networks"""

    def __init__(
            self,
            input_shape,  # State dimensions
            action_dims,  # Tuple of action dimensions (e.g., (rows, cols, 10))
            hidden_dim=128,  # Size of hidden layers
            use_cnn=False  # Whether to use CNN for state processing
    ):
        self.input_shape = input_shape
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim
        self.use_cnn = use_cnn


class CriticNetwork(nn.Module):
    def __init__(self, config):
        """
        Initialize a critic network with given configuration

        Args:
            config: CriticConfig object containing network parameters
        """
        super(CriticNetwork, self).__init__()
        self.config = config
        self.action_dims = config.action_dims
        self.use_cnn = config.use_cnn
        self.hidden_dim = config.hidden_dim
        self.input_shape = config.input_shape

        # Calculate action encoding dimension
        if isinstance(self.action_dims, tuple):
            self.action_encoding_dim = sum(self.action_dims)
        else:
            self.action_encoding_dim = self.action_dims

        # CNN layers for state processing if enabled
        if self.use_cnn:
            # Assuming input_shape is (C, H, W)
            self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

            # Calculate flattened dimension after convolutions
            conv_output_size = self._get_conv_output_size(self.input_shape)

            # State processing layers
            self.fc1 = nn.Linear(conv_output_size + self.action_encoding_dim, self.hidden_dim)
        else:
            # If not using CNN, input is flattened state vector
            state_dim = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape
            self.fc1 = nn.Linear(state_dim + self.action_encoding_dim, self.hidden_dim)

        # Common layers
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)

    def _get_conv_output_size(self, shape):
        """Calculate the flattened size after convolution layers"""
        bs = 1
        x = torch.rand(bs, *shape)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        return x.flatten(1).shape[1]

    def _process_state(self, state):
        """Process state through CNN if enabled"""
        if self.use_cnn:
            x = F.relu(self.conv1(state))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.flatten(1)
            return x
        else:
            return state

    def _process_action(self, action):
        """Process action into one-hot encoded tensor"""
        if isinstance(self.action_dims, tuple) and len(self.action_dims) == 3:
            # Handle the case for 3 discrete actions as in the original code
            a1 = F.one_hot(action[:, 0], num_classes=self.action_dims[0]).float()
            a2 = F.one_hot(action[:, 1], num_classes=self.action_dims[1]).float()
            a3 = F.one_hot(action[:, 2], num_classes=self.action_dims[2]).float()
            return torch.cat([a1, a2, a3], dim=-1)
        elif isinstance(self.action_dims, tuple):
            # Handle tuple of arbitrary length
            action_encodings = [
                F.one_hot(action[:, i], num_classes=dim).float()
                for i, dim in enumerate(self.action_dims)
            ]
            return torch.cat(action_encodings, dim=-1)
        else:
            # Handle single action dimension
            return F.one_hot(action, num_classes=self.action_dims).float()

    def forward(self, state, action):
        """
        Forward pass through the critic network

        Args:
            state: Environment state tensor
            action: Action tensor

        Returns:
            Q-value of state-action pair
        """
        state = self._process_state(state)
        action_encoded = self._process_action(action)

        x = torch.cat([state, action_encoded], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class Critic(nn.Module):
    def __init__(self, config):
        """
        Initialize dual critic networks for TD3/SAC algorithms

        Args:
            config: CriticConfig object containing network parameters
        """
        super(Critic, self).__init__()
        self.Q1 = CriticNetwork(config)
        self.Q2 = CriticNetwork(config)

    def forward(self, state, action):
        """
        Forward pass through both critic networks

        Args:
            state: Environment state tensor
            action: Action tensor

        Returns:
            Tuple of Q-values from both networks
        """
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2


if __name__ == "__main__":
    # Quick check for the Critic module.
    batch_size = 2

    # For a simple vector-based state, suppose the state dimension is 10.
    # This could also be provided as a tuple (10,) if needed by the actor.
    state_dim = 10

    # Define dummy discrete action dimensions, e.g. (rows, cols, values); here, all are size 4.
    action_dims = (4, 4, 4)

    # Create config object first
    config = CriticConfig(
        input_shape=state_dim,
        action_dims=action_dims,
        hidden_dim=128,
        use_cnn=False  # Set to True if you want to use CNN
    )

    # Then initialize the critic with the config
    critic = Critic(config)

    # Create dummy state input with shape (batch_size, state_dim).
    state = torch.randn(batch_size, state_dim)

    # Create dummy discrete actions with valid integer values in the appropriate ranges.
    # Here the actions are manually chosen within valid ranges for demonstration.
    action = torch.tensor([[0, 1, 2], [3, 0, 1]], dtype=torch.long)

    # Execute a forward pass.
    q1, q2 = critic(state, action)
    print("Q1 output:", q1)
    print("Q2 output:", q2)
