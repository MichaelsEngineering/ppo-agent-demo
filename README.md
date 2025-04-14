# TorchRL Project (Work in progress) 
This repository provides a complete implementation of Soft Actor-Critic (SAC) using PyTorch and TorchRL utilities. It’s an experiment focused on demonstrating how integrating transformer architectures with SAC can capture long-term dependencies, while also navigating the increased tuning complexity inherent in such an approach.

## Project Overview

- **TorchRL-Powered**: Every component of this project leverages TorchRL. From data collection to preprocessing and model training, it’s all optimized for TorchRL workflows.
- **Rapid Experimentation**: Designed as a quick project, expect a project that is agile and experimental. It’s built for quick iterations, testing out ideas, and pushing limits.
- **Full-Cycle Demonstration**: Experience a complete pipeline including model development, evaluation, and deployment, with real emphasis on TorchRL best practices.

## Repository Structure

```plaintext
├── .idea/
├── src/
│   ├── data/
│   │   ├── simple.json 
│   └── deployment/   
│   ├── environments/
│   │   ├── environment_setup.py
│   ├── models/
│   │   ├── actor.py  
│   │   └── critic.py
│   ├── training/
│   │   ├── replay_buffer.py               
│   │   ├── train_sac.py
|   |   └── update_parameters.py
│   ├── training/
|   |   └── load_json.py
├── tests/
    └── test_model.py
├── .GITIGNORE
├── LICENSE.md
├── main.py
├── README.md
└── requirements.txt
