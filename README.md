# AeroEmbed: TabSyn Embeddings for Flight Operations Analysis

A framework for extracting and analyzing embeddings from flight operational data using TabSyn, a state-of-the-art mixed-type tabular data synthesizer. This repository demonstrates how latent representations learned during synthetic data generation can be leveraged for operational insights in Air Traffic Management (ATM).

## Overview

AeroEmbed transforms complex flight operational data (mixed categorical and numerical features) into unified vector representations that capture essential patterns and relationships. These embeddings enable various downstream applications including:

- 🌐 **Operational Pattern Discovery**: Identify airport networks, carrier signatures, and route-specific characteristics
- 🔍 **Anomaly Detection**: Detect unusual flight operations using both density-based and isolation methods
- 📊 **Cluster Analysis**: Discover natural groupings in flight operations
- 📈 **Temporal Analysis**: Analyze delay patterns, turnaround times, and seasonal variations
- 🎯 **Dimensionality Reduction**: Navigate and interpret the learned embedding space

This work is part of the SynthAIr project, focused on improving ATM automation through AI-based models.


## Repository Structure


## Installation and Setup

### System Requirements

- Python 3.10 or newer (supports up to Python 3.13)
- CUDA-compatible GPU (recommended for training CTGAN, TabSyn, and RTF models)
- At least 16GB RAM (32GB+ recommended for larger datasets)
- Sufficient disk space for storing models and synthetic datasets

### Installation

1. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/SynthAIr/AeroEmbed.git
   cd AeroEmbed
   ```

3. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

5. Verify installation:
   ```bash
   python -c "import syntabair; print('Installation successful!')"
   ```

For GPU support, ensure appropriate CUDA drivers and libraries are installed according to your hardware specifications.


## Data Preparation

## Data Preparation

### Preprocessing Flight Data

The repository includes a tool for processing flight data from sources like OAG's Flight Info Direct. This script handle:

- Data cleaning and filtering
- Feature extraction for carrier, airport, and aircraft information
- Temporal processing of scheduled and actual timestamps
- Delay and turnaround time calculation
- Train/test splitting for evaluation

To prepare flight data for analysis and modeling:

```bash
python scripts/embedding/prepare_data.py \
  --input_path data/flights.csv \
  --output_train_path data/real/train.csv \
  --output_test_path data/real/test.csv \
  --test_size 0.2 \
  --random_state 42
```

### Data Format

The processed flight data contains essential aviation-specific columns such as:
- Carrier codes and flight identifiers
- Origin and destination airports
- Aircraft type
- Scheduled and actual timestamps for departure and arrival
- Departure and arrival delays
- Turnaround time and flight duration

Additional derived features include day of week, time-of-day components, and scheduled vs. actual duration differences.


## Model Training

```bash
python scripts/train_tabsyn.py \
  --train_path data/real/train.csv \
  --model_dir models/tabsyn \
  --vae_epochs 200 \
  --diffusion_epochs 1000 \
  --embedding_dim 4 \
  --vae_lr 1e-3 \
  --diffusion_lr 3e-4 \
  --batch_size 8192 \
  --device cuda
```

## Embedding Extraction



## Analysis and Visualization
