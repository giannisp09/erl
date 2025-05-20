# Era of Experience

An implementation of the "Era of Experience" paradigm as described in "The Era of Experience" by Silver & Sutton. This project enables agents to learn continuously from their own streams of real-world interaction, with components for experience management, reward grounding, world modeling, and planning.

## Project Structure

```
.
├── configs/                # Configuration files
├── src/                    # Core implementation
│   ├── experience/         # Experience stream management
│   ├── rewards/            # Reward functions
│   ├── models/             # World models & policies
│   ├── rl/                 # RL algorithms
│   ├── planning/           # Planning modules
│   ├── training/           # Training infrastructure
│   ├── utils/              # Utilities
│   └── verifiers/          # Verifiers for RL with LLMs
├── tests/                  # Unit tests
├── examples/               # Example scripts
│   └── verifiers/          # Verifier-specific examples
└── notebooks/              # Example notebooks
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/era-of-experience.git
cd era-of-experience

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

### Running a Basic Experiment

```bash
python -m src --config-name=default
```

### Training with Custom Configuration

```bash
python -m src --config-name=experience \
    experience.buffer_size=100000 \
    training.batch_size=64
```

## Key Features

- Continuous experience stream management
- Grounded reward functions
- World model and policy networks
- Model-based planning
- PPO with intrinsic curiosity
- TensorBoard and W&B integration

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows PEP 8 guidelines. Use the following tools for code quality:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
flake8 src tests
```

## License

MIT License

## Verifiers Subpackage

The `verifiers` subpackage (located in `src/verifiers/`) provides tools, prompts, rubrics, and utilities for verifying reinforcement learning with large language models (LLMs). It includes support for training, inference, and evaluation of verifiers, as well as example scripts in `examples/verifiers/`.

- **Code:** `src/verifiers/`
- **Examples:** `examples/verifiers/`
- **Configs:** `configs/verifiers/`

## Acknowledgements

The `verifiers` subpackage is based on the original work by [William Brown](https://github.com/williambrown97/verifiers). You can find the original repository [here](https://github.com/williambrown97/verifiers). 