# ARK

## Installation

### Prerequisites

Install UV package manager:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Project Setup

1. Clone the repository and checkout the feature branch:
```bash
git clone <repository-url>
cd ARK
```

2. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

3. Download IntelliGraphs datasets:
```bash
python -m intelligraphs.data_loaders.download
```

## Project Structure

```
KGVAE-TMLR/
├── kgvae/
│   ├── model/          # Model architecture
│   │   ├── layers.py   # Transformer and attention layers
│   │   ├── models.py   # VAE models and components
│   │   └── utils.py    # Loss functions and utilities
│   └── experiments/    # Training scripts
│       └── train.py    # Main training script
├── configs/            # Configuration files
├── checkpoints/        # Model checkpoints
├── pyproject.toml      # Project dependencies
└── README.md
```

## Training

Create a configuration file in `configs/` directory and run:

```bash
python -m kgvae.experiments.train --config configs/your_config.yaml
```

### Configuration Example

```yaml
# Model architecture
encoder_type: transformer  # or 'mlp'
decoder_type: transformer  # or 'mlp'
d_model: 256
d_hidden: 512
d_latent: 128
n_layers: 4
n_heads: 8
d_ff: 1024
dropout: 0.1

# Training parameters
batch_size: 32
learning_rate: 0.001
num_epochs: 100
beta: 1.0  # KL weight

# Data parameters
dataset: FB15k-237
max_nodes: 100
max_edges: 500

# Other
experiment_name: kgvae_experiment
save_every: 10
num_workers: 4
lr_scheduler: true
```

## Weights & Biases Integration

The training script automatically logs metrics to Weights & Biases. Configure with:

```bash
python -m kgvae.experiments.train \
    --config configs/your_config.yaml \
    --wandb-project your-project-name \
    --wandb-entity your-entity
```

## Dependencies

- PyTorch >= 2.0.0
- IntelliGraphs >= 1.0.18
- Weights & Biases
- NumPy, scikit-learn, PyYAML, tqdm, matplotlib, pandas
