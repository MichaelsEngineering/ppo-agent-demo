import json
import torch
import matplotlib.pyplot as plt

def load_json_dataset(json_path="./src/data/simple.json", split="train", device=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if split not in data:
        raise ValueError(f"Split '{split}' not found in JSON data.")

    samples = data[split]
    result = []

    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict) or 'input' not in sample or 'output' not in sample:
            raise ValueError(f"Invalid sample format at index {idx}. Expected keys 'input' and 'output'.")

        # Convert JSON data to float or long, depending on your use case:
        input_tensor = torch.tensor(sample['input'], dtype=torch.float32)
        output_tensor = torch.tensor(sample['output'], dtype=torch.float32)

        # If a device is specified, move them onto that device:
        if device is not None:
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)

        result.append((input_tensor, output_tensor))

    return result

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