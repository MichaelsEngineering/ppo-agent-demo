import pytest
import numpy as np
from gymnasium.spaces import MultiBinary
from environment_setup import MatrixEnv  # Adjust module name if necessary


def test_default_initialization():
    """Test that the default environment initializes correctly with a 2x2 matrix."""
    env = MatrixEnv()

    # Reset the environment to get the initial state
    initial_state, _ = env.reset()

    # Verify that the initial state's shape matches the environment size.
    assert initial_state.shape == env.size, "Initial state shape does not match the specified size."

    # Check that the observation space is of type MultiBinary.
    assert isinstance(env.observation_space, MultiBinary), "Observation space must be a MultiBinary space."

    # Verify that the initial state contains only binary values (0 or 1).
    assert np.all(np.isin(initial_state, [0, 1])), "Initial state must contain only 0 or 1."


def test_invalid_size_too_small():
    """Test that a matrix size below 2x2 raises a ValueError."""
    with pytest.raises(ValueError, match="Matrix size must be at least 2x2"):
        MatrixEnv(size=(1, 2))


def test_invalid_size_too_large():
    """Test that a matrix size above 30x30 raises a ValueError."""
    with pytest.raises(ValueError, match="Matrix size cannot exceed 30x30"):
        MatrixEnv(size=(31, 2))
