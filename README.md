# Weekend TorchRL Project  
This repository provides a complete implementation of Soft Actor-Critic (SAC) using PyTorch and TorchRL utilities. It’s a weekend experiment focused on demonstrating how integrating transformer architectures with SAC can capture long-term dependencies, while also navigating the increased tuning complexity inherent in such an approach.

## Project Overview

- **TorchRL-Powered**: Every component of this project leverages TorchRL. From data collection to preprocessing and model training, it’s all optimized for TorchRL workflows.
- **Rapid Experimentation**: Designed as a weekend project, expect a project that is agile and experimental. It’s built for quick iterations, testing out ideas, and pushing limits.
- **Full-Cycle Demonstration**: Experience a complete pipeline including model development, evaluation, and deployment, with real emphasis on TorchRL best practices.

## Repository Structure

```plaintext
├── README.md
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── decision_transformer.py  # TorchRL implementation
│   │   └── model_utils.py
│   ├── environments/
│   │   ├── env_config.py
│   │   └── env_wrappers.py
│   ├── data/
│   │   ├── collectors.py            # TorchRL data collection
│   │   └── preprocessing.py         # TorchRL transforms
│   ├── training/
│   │   ├── trainer.py               # Using TorchRL utilities
│   │   └── evaluation.py
│   └── deployment/
│       ├── torchscript_export.py
│       ├── onnx_export.py
│       └── inference_server.py
├── notebooks/
│   ├── 01_environment_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_deployment_demo.ipynb
└── tests/
    └── test_model.py
