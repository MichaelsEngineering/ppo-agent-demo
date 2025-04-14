import torch
import mlflow
from src.utilities.load_json_dataset import load_json_dataset
from src.environments.environment_setup import MatrixEnv
from src.training.train_sac import train_sac


def main():
    # Instantiate the environment
    env = MatrixEnv(size=(2, 2))
    json_dataset_path = "./src/data/simple.json"  # Update

    # Load train and test datasets
    train_dataset = load_json_dataset(json_dataset_path, split="train", device=env.device)
    # test_dataset = load_json_dataset(json_dataset_path, split="test", device=env.device)

    # Demonstrate accessing a sample (e.g., sample 0)
    sample_input, sample_output = train_dataset[0]  # the first sample
    print("Train Dataset Sample 0 Input:\n", sample_input)
    print("Train Dataset Sample 0 Output:\n", sample_output)
    env.load_sample((sample_input, sample_output))

    mlflow.set_experiment("sac-agent-demo-2")
    mlflow.start_run(run_name="sac-run-2")

    # Make sure train_sac returns a model (a callable PyTorch model)
    actor_model = train_sac(
        env,
        num_episodes=5,
        max_steps=20,
        batch_size=32,
        buffer_capacity=1000,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        update_every=2
    )

    if actor_model is None:
        raise ValueError("train_sac did not return a valid model. Ensure that the function returns your trained model.")

    # Create a wrapper class to provide a standard predict interface for MLflow
    class ModelWrapper(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, x):
            # Check if input is a numpy array and convert to tensor if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)

            # Ensure the input is correctly shaped for the actor model
            # If it's a flat array, reshape it to match what the actor expects
            if len(x.shape) == 1:
                # Reshape based on environment's input requirements
                # This might need adjustment based on your specific environment
                x = x.reshape(1, -1)  # Add batch dimension if needed

            # Get the actor's action outputs
            with torch.no_grad():
                action_logits = self.actor(x)
                # Return the action logits as the prediction
                return action_logits

    # Create the wrapped model
    wrapped_model = ModelWrapper(actor_model)

    # Prepare input for MLflow in the correct format
    # First, ensure we have the right input shape
    # If sample_input is 2D (e.g., 2x2), flatten it to 1D
    if len(sample_input.shape) > 1:
        flattened_input = sample_input.reshape(-1)
    else:
        flattened_input = sample_input

    # Convert to numpy for MLflow
    input_example = flattened_input.detach().cpu().numpy()

    # Log the model without trying to infer a signature or provide predictions
    # This is a fallback approach when working with complex models
    mlflow.pytorch.log_model(
        wrapped_model,
        artifact_path="model",
        input_example=input_example
    )

    mlflow.end_run()


if __name__ == '__main__':
    main()
