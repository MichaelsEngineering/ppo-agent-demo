import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from typing import Tuple, Dict, Any

class MatrixEnv(gym.Env):
    """
    A custom Gymnasium environment where the state is a matrix with binary values (0 or 1).
    Each cell in the matrix represents a color: 0 for black, 1 for white.
    """

    metadata = {"render.modes": ["human"]}  # Enable future rendering support

    def __init__(self, size: Tuple[int, int] = (2, 2)):
        super(MatrixEnv, self).__init__()

        # Validate the size
        if not isinstance(size, tuple) or len(size) != 2:
            raise ValueError("Size must be a tuple of two integers")
        if size[0] < 2 or size[1] < 2:
            raise ValueError("Matrix size must be at least 2x2")
        if size[0] > 30 or size[1] > 30:
            raise ValueError("Matrix size cannot exceed 30x30")

        self.size = size

        # Define the observation space:
        # We use gymnasium.spaces.MultiBinary with shape (size[0], size[1]) to represent a binary matrix.
        self.observation_space = spaces.MultiBinary(self.size)

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """
        Reset the environment to an initial state and return the observation.

        Returns:
            observation (dict): A dictionary with two keys:
                - "current": the grid the agent modifies.
                - "target": the grid the agent is trying to match.
            info (dict): An empty dictionary in this example.
        """
        # Set the random seed if provided.
        if seed is not None:
            np.random.seed(seed)

        # For demonstration, assume the target grid is randomly generated using binary values (0 or 1)
        # but later this can extend to values 0-9.
        self.target = np.random.randint(0, 2, size=self.observation_space.shape)

        # Initialize the current grid; you might want to start with zeros or a random configuration.
        self.current = np.zeros(self.observation_space.shape, dtype=int)

        # Build the observation dictionary.
        observation = {"current": self.current, "target": self.target}
        return observation, {}

    def step(self, action: tuple) -> tuple:
        """
        Execute one time step within the environment with a flexible cell update.

        Parameters:
            action (tuple): A tuple (i, j, new_value) where:
                - i (int): Row index of the cell to update.
                - j (int): Column index of the cell to update.
                - new_value (int): The new value to set at cell (i, j); must be an integer between 0 and 9.

        Returns:
            observation (dict): A dictionary containing:
                - "current": The updated current grid (np.ndarray).
                - "target": The target grid that the agent is attempting to match.
            reward (float): 1.0 if the update for the selected cell matches the target grid’s value at (i, j); 0.0 otherwise.
            terminated (bool): True if the entire current grid matches the target grid; otherwise, False.
            truncated (bool): Always False (truncation not implemented).
            info (dict): Additional diagnostic information; for example, the total count of cells that currently match the target.
        """
        # Validate the action tuple (i, j, new_value).
        if not (isinstance(action, (tuple, list)) and len(action) == 3):
            raise ValueError("Action must be a tuple (i, j, new_value).")

        i, j, new_value = action

        # Validate that the cell coordinates are within bounds.
        if not (0 <= i < self.current.shape[0] and 0 <= j < self.current.shape[1]):
            raise ValueError("Action coordinates are out of bounds.")

        # Validate new_value, now allowing both Python ints and NumPy integers.
        if not (isinstance(new_value, (int, np.integer)) and 0 <= new_value <= 9):
            raise ValueError("New value must be an integer between 0 and 9.")

        # Update the cell (single cell update).
        self.current[i, j] = new_value

        # Compute reward: focus on matching the target at the cell updated.
        reward = 1.0 if self.current[i, j] == self.target[i, j] else 0.0

        # Termination: when the current grid fully matches the target grid.
        terminated = bool(np.array_equal(self.current, self.target))
        truncated = False

        # Diagnostic info: count of cells that match the target.
        matching_cells = int(np.sum(self.current == self.target))
        info = {"matching_cells": matching_cells}

        # Return a dictionary observation containing both grids.
        observation = {"current": self.current, "target": self.target}

        return observation, reward, terminated, truncated, info


# --- Quick Test ---
if __name__ == "__main__":
    # Instantiate the custom environment with default size (2x2).
    env = MatrixEnv()

    # Reset to initialize the dual-grid state.
    # Now, reset returns a dictionary containing both the current and target grids.
    obs, _ = env.reset(seed=42)
    print("Initial current state:\n", obs["current"])
    print("Initial target state:\n", obs["target"])

    # Define an action: update cell (0, 0) with the target value at (0, 0).
    # This will reward the update if the value matches the target.
    new_value = obs["target"][0, 0]
    action = (0, 0, new_value)

    # Apply the action.
    obs, reward, terminated, truncated, info = env.step(action)

    print("\nAfter applying action", action)
    print("New current state:\n", obs["current"])
    print("Target state (unchanged):\n", obs["target"])
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Info:", info)
