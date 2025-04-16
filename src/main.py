import torch
import mlflow
import os

# Import the necessary elements from train_sac, including hyperparameters
from training.train_sac import (
    train_sac, process_state,
    ALPHA, GAMMA, TAU, BATCH_SIZE, LR_ACTOR, LR_CRITIC,
    MEMORY_SIZE, UPDATES_PER_STEP, START_STEPS, HIDDEN_DIM, SEED # Import necessary hyperparameters
)

from models.actor import ActorConfig
from models.critic import CriticConfig
from environments.environment_setup import MatrixEnv  # Ensure this reflects your actual env module.
from utilities.load_json_dataset import load_json_dataset


def get_action_dimensions(action_space):
    """
    Extract the dimensions of each discrete action space component.

    Args:
        action_space: A list or tuple of Discrete gym spaces

    Returns:
        tuple: Integer dimensions for each discrete action component
               For example, if action_space is (Discrete(4), Discrete(3), Discrete(5)),
               the function returns (4, 3, 5).
    """
    return tuple(space.n for space in action_space)


def main():
    # Set the MLflow tracking URI and start a run.
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("sac-agent-demos")
    # Set the computing device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset information (if required)
    ds_list = load_json_dataset(
        json_path=["./src/data/simple.json"],
        split="train",
        device=torch.device("cpu"),
        return_pytorch_dataset=False
    )
    print(f"Loaded list of pairs:\n{ds_list}")

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

    hyperparams = {
        "ALPHA": ALPHA,
        "GAMMA": GAMMA,
        "TAU": TAU,
        "BATCH_SIZE": BATCH_SIZE,
        "LR_ACTOR": LR_ACTOR,
        "LR_CRITIC": LR_CRITIC,
        "MEMORY_SIZE": MEMORY_SIZE,
        "UPDATES_PER_STEP": UPDATES_PER_STEP,
        "START_STEPS": START_STEPS,
        "HIDDEN_DIM": HIDDEN_DIM,
        "SEED": SEED
        # Add any other hyperparameters you want to track
    }

    with mlflow.start_run(run_name="sac_run_002"):
        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Train the agent using SAC.
        # Note: Pass additional training parameters to train_sac if required.
        # Pass the environment and configurations to train_sac
        trained_actor = train_sac(
            env,
            critic_config=critic_config,
            actor_config=actor_config
        )

        # Check, log training metrics.

        # Error fix:
        exact_pip_requirements = [
        f"torch=={torch.__version__}", # Dynamically get the installed torch version including local label
            # Add other essential direct dependencies here
        ]
        # Define the path to your source code directory
        # This assumes main.py is directly inside 'src'
        script_dir = os.path.dirname(__file__)
        # Correctly navigate up one level from 'src' to get the project root, then go into 'src'
        code_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
        if not os.path.exists(code_dir) or not os.path.isdir(code_dir):
            # Fallback if the structure is unexpected or script run differently
            code_dir = "./src"
            print(f"Warning: Using fallback code_path: {code_dir}")
        else:
            print(f"Using code_path: {code_dir}")  # Add print statement for verification

            # Log the trained actor model to MLflow.
        mlflow.pytorch.log_model( pytorch_model=trained_actor,
            artifact_path="actor_model",
            pip_requirements=exact_pip_requirements
        )
        print("Training complete. Model has been logged to MLflow.")


if __name__ == "__main__":
    main()
