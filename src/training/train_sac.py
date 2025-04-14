import mlflow
import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym  # in case you want to use gym wrappers, though MatrixEnv is custom
from src.models.actor import Actor
from src.models.critic import Critic
from src.training.replay_buffer import ReplayBuffer
from src.training.update_parameters import update_parameters
from mlflow.models.signature import infer_signature

def soft_update(target, source, tau):
    """
    Perform soft update of target network parameters.
    θ_target = τ*θ_source + (1 - τ)*θ_target

    Args:
        target: target network
        source: source network
        tau: interpolation parameter
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def train_sac(env, num_episodes=500, max_steps=50, batch_size=64,
              buffer_capacity=10000, gamma=0.99, tau=0.005, alpha=0.2,
              actor_lr=3e-4, critic_lr=3e-4, update_every=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine state dimension.
    # We assume env.reset() returns a dict with "current" and "target" (both flattened).
    sample_obs, _ = env.reset()
    # For simplicity, we concatenate the two tensors.
    state_sample = torch.cat([sample_obs["current"].float(), sample_obs["target"].float()], dim=0)
    state_dim = state_sample.shape[0]

    # Define action dimensions using your environment info.
    # e.g., if env.size = (rows, cols) and new_value from 0 to 9.
    rows, cols = env.size
    action_dims = (rows, cols, 10)

    # Initialize actor, critic and target critic networks.
    actor = Actor(state_dim, action_dims).to(device)
    critic = Critic(state_dim, action_dims).to(device)
    target_critic = Critic(state_dim, action_dims).to(device)
    # Copy critic weights to target critic.
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    buffer = ReplayBuffer(buffer_capacity)

    # Log hyperparameters to MLflow (assuming run was started in main.py).
    mlflow.log_param("num_episodes", num_episodes)
    mlflow.log_param("max_steps", max_steps)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("buffer_capacity", buffer_capacity)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("tau", tau)
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("actor_lr", actor_lr)
    mlflow.log_param("critic_lr", critic_lr)

    total_steps = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = torch.cat([obs["current"].float(), obs["target"].float()], dim=0)
        episode_reward = 0

        for step in range(max_steps):
            total_steps += 1
            state_tensor = state.unsqueeze(0).to(device)
            with torch.no_grad():
                action_tensor, log_prob, _ = actor.get_action(state_tensor)
            action_np = action_tensor.cpu().numpy()[0]
            action_tuple = tuple(action_np.tolist())

            next_obs, reward, terminated, truncated, info = env.step(action_tuple)
            done = terminated or truncated

            next_state = torch.cat([next_obs["current"].float(), next_obs["target"].float()], dim=0)
            episode_reward += reward

            # Store transition in the replay buffer.
            buffer.push(state.cpu().numpy(), action_np, reward, next_state.cpu().numpy(), done)
            state = next_state

            # Update network parameters periodically.
            if len(buffer) >= batch_size and total_steps % update_every == 0:
                batch = buffer.sample(batch_size)
                critic_loss, actor_loss = update_parameters(
                    batch, device, actor, critic, target_critic,
                    actor_optimizer, critic_optimizer, gamma, alpha
                )
                mlflow.log_metric("critic_loss", critic_loss, step=total_steps)
                mlflow.log_metric("actor_loss", actor_loss, step=total_steps)

            # Soft update target networks.
            soft_update(target_critic.Q1, critic.Q1, tau)
            soft_update(target_critic.Q2, critic.Q2, tau)

            if done:
                break

        mlflow.log_metric("episode_reward", episode_reward, step=episode)
        print(f"Episode {episode}: Reward {episode_reward}")

    # Suppose your actor expects shape [1, state_dim].
    example_input = torch.rand(1, state_dim).to(device)

    # Generate a sample output using the same shape
    sample_output = actor(example_input)[0].detach().cpu().numpy()

    # Build the signature
    signature = infer_signature(
        model_input=example_input.detach().cpu().numpy(),
        model_output=sample_output
    )

    # Log the actor with the signature and input example
    mlflow.pytorch.log_model(
        actor,
        artifact_path="actor_model",
        signature=signature,
        input_example=example_input.detach().cpu().numpy()
    )

    # And similarly for the critic if desired
    mlflow.pytorch.log_model(
        critic,
        artifact_path="critic_model"
        # you can define a separate example for the critic, if needed
    )
    return actor

