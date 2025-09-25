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
git clone -----Anonimized-----
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

## Training

Create a configuration file in `configs/` directory and run:

```bash
python -m kgvae.experiments.train --config configs/your_config.yaml
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
