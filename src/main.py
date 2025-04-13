import torch
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os
import shutil
from utilities.load_json_dataset import load_json_dataset
from environments.environment_setup import MatrixEnv

def save_tensor_image(tensor, title, filename):
    """
    Saves a grayscale image from a 2D tensor to disk using matplotlib.
    """
    fig, ax = plt.subplots()
    ax.imshow(tensor.cpu().numpy(), cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def format_dataset_for_humans(name, dataset):
    """
    Create a readable multiline string for visual inspection in MLflow.
    """
    lines = [f"{name} Dataset:"]
    for idx, (inp, out) in enumerate(dataset):
        lines.append(f"Sample {idx}:")
        lines.append(f"Input:\n{inp}")
        lines.append(f"Output:\n{out}")
        lines.append("")  # blank line between samples
    return "\n".join(lines)

if __name__ == "__main__":
    # Instantiate the environment
    env = MatrixEnv(size=(2, 2))
    json_dataset_path = "./src/data/simple.json" # Update

    # Load train and test datasets
    train_dataset = load_json_dataset(json_dataset_path, split="train", device=env.device)
    test_dataset = load_json_dataset(json_dataset_path, split="test", device=env.device)

    # Demonstrate accessing a sample (e.g., sample 0)
    sample_input, sample_output = train_dataset[0]  # the first sample
    print("Train Dataset Sample 0 Input:\n", sample_input)
    print("Train Dataset Sample 0 Output:\n", sample_output)

    # If your environment strictly needs them in long format, you can cast them here
    # — or, as a next step, do that inside `load_json_dataset(...)`.
    # sample_input = sample_input.long()
    # sample_output = sample_output.long()

    env.load_sample((sample_input, sample_output))

    # 5) Build an observation dict manually, just for viewing.
    observation = {
        "current": env.current.clone(),
        "target": env.target.clone()
    }
    print("Environment loaded from sample 0:", observation)

    # 6) (Optional) Demonstrate a step. For example, try to match
    #    the top-left cell in 'current' to the top-left cell in 'target'.
    i, j = 0, 0
    desired_value = int(env.target[i, j].item())
    action = (i, j, desired_value)
    obs, reward, terminated, truncated, info = env.step(action)

    print("\nAfter one step using action:", action)
    print("Obs:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)
    # Start MLflow experiment
    mlflow.set_experiment("ppo-agent-demo")

    with mlflow.start_run(run_name="dataset-logging"):
        os.makedirs("tmp_images", exist_ok=True)

        # Log train samples as images
        for idx, (inp, out) in enumerate(train_dataset):
            inp_path = f"tmp_images/train_input_{idx}.png"
            out_path = f"tmp_images/train_output_{idx}.png"
            save_tensor_image(inp, f"Train Input {idx}", inp_path)
            save_tensor_image(out, f"Train Output {idx}", out_path)
            mlflow.log_artifact(inp_path, artifact_path="train")
            mlflow.log_artifact(out_path, artifact_path="train")

        # Log test samples as images
        for idx, (inp, out) in enumerate(test_dataset):
            inp_path = f"tmp_images/test_input_{idx}.png"
            out_path = f"tmp_images/test_output_{idx}.png"
            save_tensor_image(inp, f"Test Input {idx}", inp_path)
            save_tensor_image(out, f"Test Output {idx}", out_path)
            mlflow.log_artifact(inp_path, artifact_path="test")
            mlflow.log_artifact(out_path, artifact_path="test")

        # Format and log human-readable output
        train_log = format_dataset_for_humans("Train", train_dataset)
        test_log = format_dataset_for_humans("Test", test_dataset)

        log_path = "tmp_images/train_test_output.txt"
        with open(log_path, "w") as f:
            f.write(train_log)
            f.write("\n\n")
            f.write(test_log)

        mlflow.log_artifact(log_path, artifact_path="logs")

    # Clean up temp directory
    shutil.rmtree("tmp_images")
