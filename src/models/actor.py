import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dims, hidden_dim=128):
        """
        state_dim: dimension of the flattened state
        action_dims: tuple with the number of discrete actions for each component (e.g. (rows, cols, 10))
        """
        super(Actor, self).__init__()
        self.action_dims = action_dims

        # Common shared network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Separate heads for each action component:
        self.head_i = nn.Linear(hidden_dim, action_dims[0])
        self.head_j = nn.Linear(hidden_dim, action_dims[1])
        self.head_v = nn.Linear(hidden_dim, action_dims[2])  # new_value

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits_i = self.head_i(x)
        logits_j = self.head_j(x)
        logits_v = self.head_v(x)
        return logits_i, logits_j, logits_v

    def get_action(self, state):
        """
        Given a state (batched or single), sample an action and return:
         - action: a tensor of shape (batch, 3)
         - log_prob: sum of log-probabilities for each sub-action
         - distributions: the individual categorical distributions (for debugging/analysis)
        """
        logits_i, logits_j, logits_v = self.forward(state)
        # Compute probabilities
        prob_i = F.softmax(logits_i, dim=-1)
        prob_j = F.softmax(logits_j, dim=-1)
        prob_v = F.softmax(logits_v, dim=-1)
        # Form categorical distributions
        dist_i = torch.distributions.Categorical(prob_i)
        dist_j = torch.distributions.Categorical(prob_j)
        dist_v = torch.distributions.Categorical(prob_v)
        # Sample action parts
        action_i = dist_i.sample()
        action_j = dist_j.sample()
        action_v = dist_v.sample()
        # Sum of log probabilities
        log_prob = dist_i.log_prob(action_i) + dist_j.log_prob(action_j) + dist_v.log_prob(action_v)
        # Combine parts into a (batch, 3) tensor
        action = torch.stack([action_i, action_j, action_v], dim=-1)
        return action, log_prob, (dist_i, dist_j, dist_v)
