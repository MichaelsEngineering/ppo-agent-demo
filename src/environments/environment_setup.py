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

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state and returns the initial observation."""
        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Initialize a random binary matrix as the initial state
        self.state = np.random.randint(0, 2, size=self.size)

        # Return the initial state and an empty info dictionary
        return self.state, {}


# --- Quick Test ---
if __name__ == "__main__":
    # Instantiate the custom environment with default size (2x2).
    env_default = MatrixEnv()

    # Test invalid sizes
    try:
        MatrixEnv(size=(1, 2))
    except ValueError as e:
        print("Error:", e)

    try:
        MatrixEnv(size=(31, 2))
    except ValueError as e:
        print("Error:", e)
