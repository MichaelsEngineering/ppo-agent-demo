import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dims, hidden_dim=128):
        """
        state_dim: dimension of the state vector
        action_dims: tuple as above (e.g. (rows, cols, 10))
        """
        super(CriticNetwork, self).__init__()
        self.action_dims = action_dims
        input_dim = state_dim + sum(action_dims)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # action is expected to be a tensor of shape (batch, 3) of ints.
        a1 = F.one_hot(action[:, 0], num_classes=self.action_dims[0]).float()
        a2 = F.one_hot(action[:, 1], num_classes=self.action_dims[1]).float()
        a3 = F.one_hot(action[:, 2], num_classes=self.action_dims[2]).float()
        a = torch.cat([a1, a2, a3], dim=-1)
        x = torch.cat([state, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class Critic(nn.Module):
    def __init__(self, state_dim, action_dims, hidden_dim=128):
        super(Critic, self).__init__()
        self.Q1 = CriticNetwork(state_dim, action_dims, hidden_dim)
        self.Q2 = CriticNetwork(state_dim, action_dims, hidden_dim)

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2
