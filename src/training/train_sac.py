import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
from src.models.actor import Actor, ActorConfig
from src.models.critic import Critic, CriticConfig
from src.environments.environment_setup import MatrixEnv
from collections import deque
import random
import mlflow

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
ALPHA = 0.2  # Temperature parameter for entropy
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Soft update parameter
BATCH_SIZE = 128
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
MEMORY_SIZE = 100000
UPDATES_PER_STEP = 1
START_STEPS = 10000
HIDDEN_DIM = 128

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def soft_update(target, source, tau):
    """
    Perform soft update of target network parameters

    Args:
        target: Target network
        source: Source network
        tau: Interpolation parameter (0-1)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def process_state(state_dict):
    """
    Process environment state dictionary into tensors

    Args:
        state_dict: Dictionary with 'current' and 'target' keys

    Returns:
        Processed state tensor
    """
    current = state_dict['current']
    target = state_dict['target']

    # Ensure tensors are on the correct device
    if not isinstance(current, torch.Tensor):
        current = torch.tensor(current, device=device)
    else:
        current = current.to(device)

    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, device=device)
    else:
        target = target.to(device)

    # Combine current and target into a single state representation
    state = torch.cat((current, target), dim=0)
    return state


def train_sac(env, critic_config=None, actor_config=None, max_episodes=100, max_steps=50, verbose=True):
    """
    Train an agent using Soft Actor-Critic algorithm

    Args:
        env: Gymnasium environment
        critic_config: Configuration for critic networks
        actor_config: Configuration for actor network
        max_episodes: Maximum number of episodes to train
        max_steps: Maximum steps per episode
        verbose: Whether to print progress

    Returns:
        Trained actor network
    """
    # Initialize replay buffer
    replay_buffer = deque(maxlen=MEMORY_SIZE)

    # Get state and action dimensions from environment
    state_sample, _ = env.reset()
    state = process_state(state_sample)
    state_dim = state.shape[0]

    # Get action space dimensions
    action_space = env.action_space

    # Create configurations if not provided
    if critic_config is None:
        critic_config = CriticConfig(
            input_shape=state_dim,
            action_dims=action_space,
            hidden_dim=HIDDEN_DIM,
            use_cnn=False
        )

    if actor_config is None:
        # Assuming ActorConfig has similar structure to CriticConfig
        actor_config = ActorConfig(
            input_shape=state_dim,
            action_dims=action_space,
            hidden_dim=HIDDEN_DIM,
            use_cnn=False
        )

    # Initialize actor and critics
    actor = Actor(actor_config).to(device)
    critic = Critic(critic_config).to(device)
    critic_target = Critic(critic_config).to(device)

    # Copy parameters from critic to critic_target
    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)

    # Initialize optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # Training metrics
    total_steps = 0
    best_reward = -float('inf')
    rewards_history = []
    losses_history = {'actor': [], 'critic': [], 'alpha': []}

    # Training loop
    for episode in range(max_episodes):
        episode_reward = 0
        episode_steps = 0
        done = False
        state_dict, _ = env.reset()
        state = process_state(state_dict)

        # Episode loop
        while not done and episode_steps < max_steps:
            # Select action
            if total_steps < START_STEPS:
                # Random action for exploration
                action_i = np.random.randint(0, action_space[0].n)
                action_j = np.random.randint(0, action_space[1].n)
                action_val = np.random.randint(0, action_space[2].n)
                action = (action_i, action_j, action_val)
            else:
                # Use actor for action selection
                with torch.no_grad():
                    action = actor.select_action(state)

            # Take action in environment
            next_state_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = process_state(next_state_dict)

            # Store transition in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))

            # Update statistics
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Update networks
            if len(replay_buffer) > BATCH_SIZE and total_steps % UPDATES_PER_STEP == 0:
                for _ in range(UPDATES_PER_STEP):
                    # Sample mini-batch
                    batch = random.sample(replay_buffer, BATCH_SIZE)
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

                    # Convert to tensors
                    state_batch = torch.stack(
                        [s if isinstance(s, torch.Tensor) else torch.tensor(s, device=device) for s in state_batch]).to(
                        device)
                    next_state_batch = torch.stack(
                        [s if isinstance(s, torch.Tensor) else torch.tensor(s, device=device) for s in
                         next_state_batch]).to(device)
                    reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(device)
                    done_batch = torch.tensor(done_batch, dtype=torch.float).to(device)

                    # Process actions (converting from tuples to appropriate format)
                    action_tensor_list = []
                    for a in action_batch:
                        action_i, action_j, action_val = a
                        action_tensor = torch.tensor([action_i, action_j, action_val], dtype=torch.long).to(device)
                        action_tensor_list.append(action_tensor)
                    action_batch = torch.stack(action_tensor_list).to(device)

                    # Critic update
                    with torch.no_grad():
                        next_actions, next_log_probs = actor.sample_with_log_prob(next_state_batch)
                        target_q1, target_q2 = critic_target(next_state_batch, next_actions)

                        # <<< Add shape printing for components >>>
                        print(f"\n--- Target Q Calculation ---")
                        print(f"Shape - next_state_batch: {next_state_batch.shape}")
                        # Check if next_actions is a tensor before printing shape
                        if isinstance(next_actions, torch.Tensor):
                            print(f"Shape - next_actions: {next_actions.shape}")
                        else:
                            print(f"Type - next_actions: {type(next_actions)}")  # Print type if not tensor
                        print(f"Shape - target_q1 (from target critic): {target_q1.shape}")
                        print(f"Shape - target_q2 (from target critic): {target_q2.shape}")
                        print(f"Shape - next_log_probs (from actor): {next_log_probs.shape}")
                        # <<< End added shape printing >>>

                        # Ensure next_log_probs has the correct shape [batch_size, 1] for subtraction
                        if next_log_probs.ndim == 1:
                            print(
                                f"Reshaping next_log_probs from {next_log_probs.shape} to {next_log_probs.unsqueeze(1).shape}")
                            next_log_probs = next_log_probs.unsqueeze(1)
                        # Also ensure target_q1/q2 are [batch_size, 1] before min()
                        if target_q1.shape != target_q2.shape or target_q1.ndim != 2 or target_q1.shape[1] != 1:
                            print(
                                f"Warning: Unexpected shape for target_q1 ({target_q1.shape}) or target_q2 ({target_q2.shape}) before torch.min")

                        # Calculate intermediate target Q value (before Bellman update)
                        target_q_intermediate = torch.min(target_q1, target_q2) - ALPHA * next_log_probs

                        # <<< Add shape printing for intermediate target_q >>>
                        print(f"Shape - target_q (after min and entropy subtraction): {target_q_intermediate.shape}")
                        # <<< End added shape printing >>>

                        # Final Bellman update
                        # Ensure reward_batch and done_batch are correctly shaped ([batch_size, 1])
                        reward_reshaped = reward_batch.unsqueeze(1)
                        done_reshaped = done_batch.unsqueeze(1)
                        if reward_reshaped.shape[0] != BATCH_SIZE or done_reshaped.shape[0] != BATCH_SIZE:
                            print(
                                f"Warning: Shape mismatch in reward ({reward_reshaped.shape}) or done ({done_reshaped.shape}) batch before Bellman update")

                        target_q = reward_reshaped + GAMMA * (1 - done_reshaped) * target_q_intermediate

                        # <<< Add shape printing for final target_q before loss >>>
                        print(f"Shape - target_q (final before loss): {target_q.shape}")
                        print(f"--- End Target Q Calculation ---\n")
                        # <<< End added shape printing >>>

                    current_q1, current_q2 = critic(state_batch, action_batch)
                    # <<< Add shape checks right here >>>
                    print(f"Shape Check JUST BEFORE F.mse_loss:")
                    print(f"  Shape - current_q1: {current_q1.shape}")
                    print(f"  Shape - current_q2: {current_q2.shape}")
                    print(f"  Shape - target_q:   {target_q.shape}") # Verify this one again here
                    # <<< End added shape checks >>>


                    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    # Actor update
                    actions, log_probs = actor.sample_with_log_prob(state_batch)
                    q1, q2 = critic(state_batch, actions)
                    q = torch.min(q1, q2)
                    actor_loss = (ALPHA * log_probs - q).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Soft update of target network
                    soft_update(critic_target, critic, TAU)

                    # Record losses
                    losses_history['critic'].append(critic_loss.item())
                    losses_history['actor'].append(actor_loss.item())

        # End of episode
        rewards_history.append(episode_reward)

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(actor.state_dict(), 'src/models/best_actor.pt')

        # Log progress
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}/{max_episodes} | Avg Reward: {avg_reward:.2f} | Steps: {total_steps}")

    # Save final model
    torch.save(actor.state_dict(), 'src/models/final_actor.pt')

    # Save training metrics
    metrics = {
        'rewards': rewards_history,
        'losses': losses_history
    }

    # Log metrics and model conditionally
    if mlflow.active_run():
        # MLflow is active: log metrics and model
        print("MLflow run active. Logging metrics and model to MLflow.")
        for episode, reward in enumerate(rewards_history):
            mlflow.log_metric("reward", reward, step=episode)

        # Log losses if available
        if losses_history.get('actor'):
            for step, actor_loss in enumerate(losses_history['actor']):
                mlflow.log_metric("actor_loss", actor_loss, step=step)
                # Ensure critic loss exists for the same step
                if step < len(losses_history.get('critic', [])):
                    mlflow.log_metric("critic_loss", losses_history['critic'][step], step=step)

        # Log the final actor model
        mlflow.pytorch.log_model(actor, "actor_model")
    else:
        # MLflow is not active: save metrics to JSON
        print("No active MLflow run. Saving metrics to training_metrics.json.")
        metrics = {
            'rewards': rewards_history,
            'losses': losses_history
        }
        try:
            # Save training metrics to JSON file
            with open('training_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)  # Added indent for better readability
            print("Metrics successfully saved to training_metrics.json")
        except Exception as e:
            print(f"Error saving metrics to JSON: {e}")

    return actor


if __name__ == "__main__":
    env = MatrixEnv(size=(2, 2), device="cpu")

    # Get environment details for model initialization
    state_sample, _ = env.reset()
    state = process_state(state_sample)
    state_dim = state.shape[0]

    # Convert each action space to its integer dimension
    # e.g. if env.action_space is (Discrete(4), Discrete(3), Discrete(5)),
    #      then action_dims becomes (4, 3, 5).
    action_dims = tuple(space.n for space in env.action_space)

    # Create a configuration for the critic
    critic_config = CriticConfig(
        input_shape=(state_dim,),
        action_dims=action_dims,
        hidden_dim=128,
        use_cnn=False
    )

    # Create a configuration for the actor
    actor_config = ActorConfig(
        input_shape=(state_dim,),
        action_dims=action_dims,
        hidden_dim=128,
        use_cnn=False
    )

    # Pass the environment and configurations to train_sac
    trained_actor = train_sac(
        env,
        critic_config=critic_config,
        actor_config=actor_config
    )



