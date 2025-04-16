import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List
import mlflow
import mlflow.pytorch


@dataclass
class ActorConfig:
    """Configuration for Actor neural network.

    Attributes:
        input_shape: Shape of the input observation (channels, height, width) for images
                     or (feature_dim,) for vector observations.
        action_dims: Tuple specifying the size of each discrete action component, e.g. (n_i, n_j, n_v).
        hidden_dim: Number of hidden units in the fully connected layers.
        use_cnn: Whether to use convolutional layers (True) for image input or
                 fully connected layers (False) for vector input.
    """
    input_shape: Tuple[int, ...]
    action_dims: Tuple[int, ...]
    hidden_dim: int = 128
    use_cnn: bool = False


class Actor(nn.Module):
    """Actor network for reinforcement learning with discrete action spaces.

    This network processes image observations with CNN layers (when use_cnn=True)
    or vector observations with fully connected layers (when use_cnn=False).
    It outputs separate logits for each action component in discrete action spaces.
    """

    def __init__(self, config: ActorConfig):
        """
        Args:
            config: An ActorConfig dataclass instance with network hyperparameters.
        """
        super().__init__()
        self.input_shape = config.input_shape
        self.action_dims = config.action_dims
        self.hidden_dim = config.hidden_dim
        self.use_cnn = config.use_cnn

        if self.use_cnn:
            # For image-based input (channels, height, width)
            self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

            # Calculate expected output size after CNN layers based on the configured input size.
            h, w = self.input_shape[1], self.input_shape[2]
            h //= 2  # After conv1 (stride=2)
            w //= 2
            h //= 2  # After conv2 (stride=2)
            w //= 2
            flat_size = 64 * h * w

            self.fc1 = nn.Linear(flat_size, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            # For vector-based input
            self.fc1 = nn.Linear(self.input_shape[0], self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Action distribution heads (logits) for each discrete action component
        self.head_i = nn.Linear(self.hidden_dim, self.action_dims[0])
        self.head_j = nn.Linear(self.hidden_dim, self.action_dims[1])
        self.head_v = nn.Linear(self.hidden_dim, self.action_dims[2])

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: The input state tensor. For CNN inputs, shape is (batch_size, channels, height, width).
                   For fully connected inputs, shape is (batch_size, feature_dim).

        Returns:
            A tuple of three tensors (logits_i, logits_j, logits_v) corresponding
            to the discrete action components.
        """
        if self.use_cnn:
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
        else:
            x = state

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits_i = self.head_i(x)
        logits_j = self.head_j(x)
        logits_v = self.head_v(x)
        return logits_i, logits_j, logits_v

    def get_action(self, state: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, List[torch.distributions.Categorical]]:
        """
        Samples an action from the Actor's policy given the input state.

        Args:
            state: The input state tensor.

        Returns:
            A tuple containing:
            - action: A tensor of shape (batch_size, 3) with the sampled discrete actions.
            - log_prob: The log probability of the sampled action (summed across components).
            - dists: A list of the categorical distributions [dist_i, dist_j, dist_v].
        """
        logits_i, logits_j, logits_v = self.forward(state)

        dist_i = torch.distributions.Categorical(logits=logits_i)
        dist_j = torch.distributions.Categorical(logits=logits_j)
        dist_v = torch.distributions.Categorical(logits=logits_v)

        action_i = dist_i.sample()
        action_j = dist_j.sample()
        action_v = dist_v.sample()

        log_prob = dist_i.log_prob(action_i) + dist_j.log_prob(action_j) + dist_v.log_prob(action_v)
        action = torch.stack([action_i, action_j, action_v], dim=-1)

        return action, log_prob, [dist_i, dist_j, dist_v]

    def sample_with_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy and return the actions along with their log probabilities.

        Args:
            state: The current state batch tensor.

        Returns:
            actions: Tensor of shape (batch_size, 3) containing the sampled discrete actions.
            log_prob: Tensor of shape (batch_size,) with the summed log probabilities of the sampled actions.
        """
        # Defensive check: Ensure the state is a float tensor.
        if state.dtype != torch.float32:
            state = state.float()

        # Forward pass: obtain logits for each discrete action component.
        logits_i, logits_j, logits_v = self.forward(state)

        # Create categorical distributions from logits.
        dist_i = torch.distributions.Categorical(logits=logits_i)
        dist_j = torch.distributions.Categorical(logits=logits_j)
        dist_v = torch.distributions.Categorical(logits=logits_v)

        # Sample actions from each distribution.
        action_i = dist_i.sample()
        action_j = dist_j.sample()
        action_v = dist_v.sample()

        # Compute log probabilities for each sampled action.
        log_prob_i = dist_i.log_prob(action_i)
        log_prob_j = dist_j.log_prob(action_j)
        log_prob_v = dist_v.log_prob(action_v)

        # Sum log probabilities for the full action.
        log_prob = log_prob_i + log_prob_j + log_prob_v

        # Combine the action components into a single tensor of shape (batch_size, 3).
        actions = torch.stack([action_i, action_j, action_v], dim=1)

        return actions, log_prob


def process_observation(obs):
    """Placeholder for any observation processing logic.

    This can include normalization and conversion to a torch tensor.
    """
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs)
    return obs.float()


def process_observation_fixed_size(obs, target_size):
    """
    Resizes the observation to the target size.

    Args:
        obs: A tensor or array of shape (C, H, W).
        target_size: A tuple (C, H_target, W_target); channels are assumed to match.

    Returns:
        A resized observation tensor of shape target_size.
    """
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs)
    # Ensure obs has shape (1, C, H, W) for interpolation.
    if obs.dim() == 3:
        obs = obs.unsqueeze(0)
    # Extract desired height and width from target_size.
    _, H_target, W_target = target_size
    resized = F.interpolate(obs, size=(H_target, W_target), mode='bilinear', align_corners=False)
    # Remove the batch dimension before returning.
    return resized.squeeze(0)


def main():
    # For image input, we expect input_shape=(3, 64, 64) and discrete action dimensions, for example, (10, 10, 10).
    input_shape = (3, 64, 64)
    action_dims = (10, 10, 10)

    # Small actor configuration with CNN enabled.
    actor_config_small = ActorConfig(
        input_shape=input_shape,
        action_dims=action_dims,
        hidden_dim=128,
        use_cnn=True
    )
    actor_small = Actor(config=actor_config_small)
    small_obs = torch.randn(1, 3, 64, 64)

    with mlflow.start_run(run_name="Actor_Small_Run"):
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.log_params({
            "input_shape": actor_config_small.input_shape,
            "action_dims": actor_config_small.action_dims,
            "hidden_dim": actor_config_small.hidden_dim,
            "use_cnn": actor_config_small.use_cnn
        })
        action_small, log_prob_small, dists_small = actor_small.get_action(small_obs)
        mlflow.log_metric("avg_log_prob", log_prob_small.mean().item())
        mlflow.pytorch.log_model(actor_small, artifact_path="actor_small_model", input_example=small_obs)

    # Large actor configuration with increased hidden dimension.
    actor_config_large = ActorConfig(
        input_shape=input_shape,
        action_dims=action_dims,
        hidden_dim=256,
        use_cnn=True
    )
    actor_large = Actor(config=actor_config_large)
    large_obs = torch.randn(1, 3, 64, 64)

    with mlflow.start_run(run_name="Actor_Large_Run"):
        mlflow.log_params({
            "input_shape": actor_config_large.input_shape,
            "action_dims": actor_config_large.action_dims,
            "hidden_dim": actor_config_large.hidden_dim,
            "use_cnn": actor_config_large.use_cnn
        })
        action_large, log_prob_large, dists_large = actor_large.get_action(large_obs)
        mlflow.log_metric("avg_log_prob", log_prob_large.mean().item())
        mlflow.pytorch.log_model(actor_large, artifact_path="actor_large_model", input_example=large_obs)

    # For a medium observation, simulate an input of size (3, 100, 100) that needs resizing.
    target_size = (3, 64, 64)
    medium_obs = torch.randn(3, 100, 100)
    # Resize the observation to the target size.
    resized_obs = process_observation_fixed_size(medium_obs, target_size)
    resized_obs = resized_obs.float().unsqueeze(0)  # shape becomes (1, 3, 64, 64)

    actor_config_standard = ActorConfig(
        input_shape=target_size,
        action_dims=action_dims,
        hidden_dim=128,
        use_cnn=True
    )
    actor_standard = Actor(config=actor_config_standard)

    with mlflow.start_run(run_name="Actor_Standard_Run"):
        mlflow.log_params({
            "input_shape": actor_config_standard.input_shape,
            "action_dims": actor_config_standard.action_dims,
            "hidden_dim": actor_config_standard.hidden_dim,
            "use_cnn": actor_config_standard.use_cnn
        })
        action_standard, log_prob_resized, dists_resized = actor_standard.get_action(resized_obs)
        mlflow.log_metric("avg_log_prob", log_prob_resized.mean().item())
        mlflow.pytorch.log_model(actor_standard, artifact_path="actor_standard_model", input_example=actor_standard)


if __name__ == "__main__":
    main()
