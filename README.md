## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap-for-contributions)
- [Contributing](#contributing)
- [License](#license)

# SAC Torch MLFLow Vibe Project

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A vibe coded implementation of Soft Actor-Critic (SAC) using PyTorch and MLFlow. This project demonstrates how transformer architectures can be integrated with SAC to effectively capture long-term dependencies in reinforcement learning tasks.

## Key Features

- **Torch-Powered**: Every component leverages Torch's optimized workflows for data collection, preprocessing, and model training
- **Transformer-Enhanced RL**: Novel integration of transformer architecture with SAC for superior temporal reasoning
- **MLflow Integration**: Complete experiment tracking with parameter logging and model versioning
- **Modular Design**: Clean separation of environment, model, and training components for easy extension

## Technical Insights
- **Rapid Development**: This implementation was developed in approximately 12 hours as a proof-of-concept, demonstrating rapid prototyping capability while maintaining a clean architecture. It showcases the ability to quickly deliver working machine learning systems.
- **Production Readiness**: While built as a rapid prototype, the codebase follows a modular design with clear separation between environment, models, and training components. If continued, future iterations will focus on implementing proper logging with configurable verbosity levels and comprehensive exception handling.


## Installation

1. Clone the repository:

   ```bash
   gh repo clone MichaelsEngineering/sac-agent-demo
   cd sac-agent-demo
   ```
   
2. Create an env and install the required dependencies:

   ```bash
   python -m venv sac-env
   source sac-env/bin/activate  # Linux/macOS
   # Windows: sac-env\Scripts\activate  
   pip install -e .
   # Or, for development, include additional dev dependencies
   pip install -e ".[dev]"
   ```
3. Track Experiments:

 ```bash
 mlflow ui # Then open in browser
 ```

4. Run main:

```bash
   python src/main.py
```

## Usage

1. Run the project
   ```bash
   python src/main.py
   ```
   
## Project Structure

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
│   ├── training/
|   |   └── load_json.py
│   └── main.py
├── tests/
│   └── test_model.py
├── .GITIGNORE
├── LICENSE.md
├── pip_requirements.txt
├── pyproject.toml
└── README.md
```


## Roadmap for Contributions
- Upgrade Data Loading for CI/CD engineering best practices
- Add support for continuous action spaces
- Enhance MLflow dashboards 
- Containerize with Docker for reproducible deployment


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)