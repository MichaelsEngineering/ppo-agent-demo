import json
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Optional



class TensorPairsDataset(Dataset):
    """
    Minimal PyTorch Dataset that holds a list of (input_tensor, output_tensor) pairs.
    """
    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs

    def __getitem__(self, idx):
        return self.pairs[idx]

    def __len__(self):
        return len(self.pairs)


def load_json_dataset(
        json_path: Union[str, List[str]] = "./src/data/simple.json",
        split: str = "train",
        device: Optional[torch.device] = None,
        random_split_ratio: Optional[float] = None,
        seed: int = 42,
        return_pytorch_dataset: bool = False
) -> Union[List[Tuple[torch.Tensor, torch.Tensor]], Tuple[List, List], TensorPairsDataset, Tuple[
    TensorPairsDataset, TensorPairsDataset]]:
    """
    Load dataset from JSON file(s) with enhanced flexibility.

    Args:
        json_path: Path to JSON file or list of paths to multiple JSON files
        split: Data split to load ('train', 'test', 'val', etc.)
        device: Device to place tensors on (CPU/GPU)
        random_split_ratio: If provided, randomly split data with this ratio (train size)
            Only used if the split is not already defined in the JSON file
        seed: Random seed for reproducibility
        return_pytorch_dataset: If True, return PyTorch Dataset object(s) instead of tensor pairs

    Returns:
        Either:
        - List of (input, output) tensor pairs
        - A tuple of (train_data, test_data) lists if random_split_ratio is used
        - A TensorPairsDataset containing the data
        - A tuple of (train_dataset, test_dataset) if random_split_ratio is used with return_pytorch_dataset=True
    """
    # Handle single file or multiple files
    if isinstance(json_path, str):
        json_paths = [json_path]
    else:
        json_paths = json_path

    all_samples = []

    # Load data from all specified JSON files
    for path in json_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        # Check if the specified split exists in the file
        if split in data:
            samples = data[split]
            all_samples.extend(samples)
        elif random_split_ratio is None and split != "all":
            raise ValueError(f"Split '{split}' not found in JSON data at {path}.")
        elif split == "all":
            # Combine all splits if 'all' is requested
            all_splits_data = []
            for available_split in data.keys():
                all_splits_data.extend(data[available_split])
            all_samples.extend(all_splits_data)

    # If no samples were found with the specified split but we have a random_split_ratio,
    # we'll try to use all available data and split it manually
    if len(all_samples) == 0 and random_split_ratio is not None:
        for path in json_paths:
            with open(path, 'r') as f:
                data = json.load(f)

            # Get all data regardless of split
            for available_split in data.keys():
                all_samples.extend(data[available_split])

    # Process the samples into tensor pairs
    result = []
    for idx, sample in enumerate(all_samples):
        if not isinstance(sample, dict) or 'input' not in sample or 'output' not in sample:
            raise ValueError(f"Invalid sample format at index {idx}. Expected keys 'input' and 'output'.")

        # Convert to tensors
        input_tensor = torch.tensor(sample['input'], dtype=torch.float32)
        output_tensor = torch.tensor(sample['output'], dtype=torch.float32)

        # Move to specified device if needed
        if device is not None:
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)

        result.append((input_tensor, output_tensor))

    # Handle random splitting if requested
    if random_split_ratio is not None:
        import random
        random.seed(seed)
        train_size = int(random_split_ratio * len(result))
        test_size = len(result) - train_size

        indices = list(range(len(result)))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_data = [result[i] for i in train_indices]
        test_data = [result[i] for i in test_indices]

        if return_pytorch_dataset:
            return TensorPairsDataset(train_data), TensorPairsDataset(test_data)
        else:
            return train_data, test_data

    # Return as PyTorch Dataset if requested
    if return_pytorch_dataset:
        return TensorPairsDataset(result)

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
    Create a readable multiline string for visual inspection in MLflow or logs.
    """
    lines = [f"{name} Dataset:"]
    for idx, (inp, out) in enumerate(dataset):
        lines.append(f"Sample {idx}:")
        lines.append(f"Input:\n{inp}")
        lines.append(f"Output:\n{out}")
        lines.append("")  # Blank line between samples
    return "\n".join(lines)



if __name__ == "__main__":
    # Example usage with the updated function
    print("Testing list of tensor pairs:")
    ds_list = load_json_dataset(
        json_path=["./src/data/simple.json"],  # changed from json_paths to json_path
        split="train",
        device=torch.device("cpu"),
        # random_split_ratio=None,  # Not needed, it's optional
        # seed=42,  # Not needed, it's optional
        return_pytorch_dataset=False  # changed from as_dataset to return_pytorch_dataset
    )
    print(f"Loaded list of pairs: \n{ds_list}")

    for idx, (input_tensor, target_tensor) in enumerate(ds_list):
        input_list = input_tensor.tolist()
        target_list = target_tensor.tolist()
        print(f"Pair {idx + 1}:\n  Input: {input_list}\n  Target: {target_list}\n")

    print("\nTesting PyTorch Dataset:")
    ds_dataset = load_json_dataset(
        json_path=["./src/data/simple.json"],  # changed from json_paths to json_path
        split="test",
        device=torch.device("cpu"),
        # random_split_ratio=None,  # Not needed, it's optional
        # seed=42,  # Not needed, it's optional
        return_pytorch_dataset=True  # changed from as_dataset to return_pytorch_dataset
    )
    print(f"Loaded as PyTorch Dataset; length = {len(ds_dataset)}")

    # Test random split functionality
    print("\nTesting random split functionality:")
    train_data, test_data = load_json_dataset(
        json_path="./src/data/simple.json",
        split="all",  # Use all data
        random_split_ratio=0.7,  # 70% train, 30% test
        return_pytorch_dataset=False
    )
    print(f"Random split created:\n  Train size: {len(train_data)}\n  Test size: {len(test_data)}")

    # Test random split with Dataset return type
    print("\nTesting random split with Dataset return type:")
    train_dataset, test_dataset = load_json_dataset(
        json_path="./src/data/simple.json",
        split="all",
        random_split_ratio=0.7,
        return_pytorch_dataset=True
    )
    print(
        f"Random split datasets:\n  Train dataset size: {len(train_dataset)}\n  Test dataset size: {len(test_dataset)}")

