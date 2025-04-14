import torch
import torch.nn.functional as F


def update_parameters(batch, device, actor, critic, target_critic,
                      actor_optimizer, critic_optimizer,
                      gamma, alpha):
    """
    Update actor and critic networks using a batch of transitions.

    batch: tuple (state, action, reward, next_state, done) as numpy arrays.
    """
    # Unpack and convert to tensors.
    states, actions, rewards, next_states, dones = batch
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # --- Critic update ---
    with torch.no_grad():
        # Get action and log probability for next state from actor.
        next_actions, next_log_probs, _ = actor.get_action(next_states)
        # Evaluate target Q-values from target networks.
        q1_target, q2_target = target_critic(next_states, next_actions)
        min_q_target = torch.min(q1_target, q2_target)
        # Compute target with entropy bonus.
        q_target = rewards + gamma * (1 - dones) * (min_q_target - alpha * next_log_probs.unsqueeze(1))

    # Current Q estimates
    current_q1, current_q2 = critic(states, actions)
    critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # --- Actor update ---
    actions_pred, log_probs, _ = actor.get_action(states)
    q1_pi, q2_pi = critic(states, actions_pred)
    min_q_pi = torch.min(q1_pi, q2_pi)
    actor_loss = (alpha * log_probs - min_q_pi.squeeze(1)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return critic_loss.item(), actor_loss.item()
