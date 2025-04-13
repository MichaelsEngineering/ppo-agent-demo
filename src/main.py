import json
import torch
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os
import shutil

def load_json_dataset(json_path="./src/data/simple.json", split="train"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if split not in data:
        raise ValueError(f"Split '{split}' not found in JSON data.")

    samples = data[split]
    result = []

    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict) or 'input' not in sample or 'output' not in sample:
            raise ValueError(f"Invalid sample format at index {idx}. Expected keys 'input' and 'output'.")

        input_tensor = torch.tensor(sample['input'], dtype=torch.float32)
        output_tensor = torch.tensor(sample['output'], dtype=torch.float32)
        result.append((input_tensor, output_tensor))

    return result

def save_tensor_image(tensor, title, filename):
    """
    Saves a grayscale image from a 2D tensor to disk using matplotlib.
    """
    fig, ax = plt.subplots()
    ax.imshow(tensor.numpy(), cmap='gray')
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
    json_dataset_path = "./src/data/simple.json"

    # Load train and test datasets
    train_dataset = load_json_dataset(json_dataset_path, split="train")
    test_dataset = load_json_dataset(json_dataset_path, split="test")

    print(train_dataset)

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
