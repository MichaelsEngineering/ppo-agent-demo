import torch
import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces
import gymnasium.utils.seeding as seeding
from typing import Tuple, Dict, Any




class MatrixEnv(gym.Env):
    """
    A custom Gymnasium environment that tracks two grids: a 'current' grid and a 'target' grid.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, size: Tuple[int, int] = (2, 2), device: str = None):
        super(MatrixEnv, self).__init__()


        # Validate the size
        if not isinstance(size, tuple) or len(size) != 2:
            raise ValueError("Size must be a tuple of two integers.")
        if size[0] < 2 or size[1] < 2:
            raise ValueError("Matrix size must be at least 2x2.")
        if size[0] > 30 or size[1] > 30:
            raise ValueError("Matrix size cannot exceed 30x30.")

        self.size = size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        discrete_dims = [10] * (size[0] * size[1])
        self.observation_space = spaces.MultiDiscrete(discrete_dims)

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.size[0]),  # i
            spaces.Discrete(self.size[1]),  # j
            spaces.Discrete(10)            # new_value in [0..9]
        ))

        # Initialize containers for current and target.
        self.current = None
        self.target = None
        self._seed_value = None

    def seed(self, seed=None):
        """
        Set the environment's random seed, and store it in self.seed_value.
        """
        if seed is not None:
            self.seed_value = seed
            torch.manual_seed(seed)
            np.random.seed(seed)

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """
        Reset the environment to an initial state and return the observation.

        Returns:
            observation (dict):
                - "current": A torch.Tensor of flattened shape (rows * cols)
                - "target": A torch.Tensor of flattened shape (rows * cols)
            info (dict): Additional info (empty here).
        """
        # We are ignoring the 'seed' to keep the signature consistent but not actually seeding anything.

        # For demonstration, we randomly generate the target grid with values 0 or 1.
        self.target = torch.randint(
            low=0,
            high=2,  # or up to 10 if you want a bigger range
            size=self.size,
            dtype=torch.long,
            device=self.device
        )

        # The current grid starts at all zeros.
        self.current = torch.zeros(
            self.size,
            dtype=torch.long,
            device=self.device
        )

        # Flatten for consistency with MultiDiscrete([10]*(rows*cols))
        observation = {
            "current": self.current.view(-1).clone(),
            "target": self.target.view(-1).clone()
        }

        return observation, {}

    def step(self, action: tuple) -> tuple:
        """
        Execute one time step within the environment.

        Args:
            action (tuple): (i, j, new_value)

        Returns:
            observation (dict): with "current" and "target" keys.
            reward (float): 1.0 if current[i, j] == target[i, j], else 0.0.
            terminated (bool): True if the entire current grid matches the target grid.
            truncated (bool): Always False here.
            info (dict): e.g. {"matching_cells": number_of_matches}
        """
        if not (isinstance(action, (tuple, list)) and len(action) == 3):
            raise ValueError("Action must be a tuple (i, j, new_value).")
        i, j, new_value = action

        # Validate that the cell coordinates are within bounds.
        if not (0 <= i < self.size[0] and 0 <= j < self.size[1]):
            raise ValueError("Action coordinates are out of bounds.")

        # Validate new_value is in [0..9].
        if not (isinstance(new_value, (int, np.integer)) and 0 <= new_value <= 9):
            raise ValueError("New value must be an integer between 0 and 9.")

        # Update the cell in the current grid.
        self.current[i, j] = new_value

        # Compute reward.
        reward = 1.0 if self.current[i, j] == self.target[i, j] else 0.0

        # Check if entire grid matches.
        terminated = bool(torch.equal(self.current, self.target))
        truncated = False

        # Diagnostic info: how many cells match the target.
        matching_cells = (self.current == self.target).sum().item()
        info = {"matching_cells": matching_cells}

        # Return the observation as a copy to avoid side-effects.
        observation = {
            "current": self.current.view(-1).clone(),
            "target": self.target.view(-1).clone()
        }

        return observation, float(reward), terminated, truncated, info

    def load_sample(self, sample):
        self.current, self.target = sample

    def render(self, mode="human"):
        """
        Render the environment's state in a human-readable way.
        """
        print("Current grid:\n", self.current.cpu().numpy())
        print("Target grid:\n", self.target.cpu().numpy())

    def close(self):
        pass


# Example usage test (if needed)
if __name__ == "__main__":
    env = MatrixEnv(size=(2, 2))
    obs, _ = env.reset(seed=42)
    print("Initial observation:", obs)
    action = (0, 0, int(obs["target"][0,0].item()))
    obs, reward, terminated, truncated, info = env.step(action)
    print("After step:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Info:", info)
