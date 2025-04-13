import json
import torch

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