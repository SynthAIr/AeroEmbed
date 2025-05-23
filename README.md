# AeroEmbed: TabSyn Embeddings for Flight Operations Analysis

A comprehensive framework for extracting and analyzing embeddings from flight operational data using TabSyn, a state-of-the-art mixed-type tabular data synthesizer. This repository demonstrates how latent representations learned during synthetic data generation can be leveraged for operational insights in Air Traffic Management (ATM).

## Overview

AeroEmbed transforms complex flight operational data (mixed categorical and numerical features) into unified vector representations that capture essential patterns and relationships. These embeddings enable various downstream applications including:

- 🌐 **Operational Pattern Discovery**: Identify airport networks, carrier signatures, and route-specific characteristics
- 🔍 **Anomaly Detection**: Detect unusual flight operations using both density-based and isolation methods
- 📊 **Cluster Analysis**: Discover natural groupings in flight operations
- 📈 **Temporal Analysis**: Analyze delay patterns, turnaround times, and seasonal variations
- 🎯 **Dimensionality Reduction**: Navigate and interpret the learned embedding space

This work is part of the SynthAIr project (Grant Agreement No. 101114847), focused on improving ATM automation through AI-based models.

## Key Features

- **Unified Embedding Space**: Handles mixed-type data (categorical and numerical) in a single representation
- **Multiple Analysis Tools**: Comprehensive suite of scripts for different analytical perspectives
- **Visualization Support**: Rich visualizations using UMAP, t-SNE, and PCA for embedding interpretation
- **Scalable Architecture**: Efficient processing of large-scale flight datasets
- **Modular Design**: Easy to extend with new analysis methods

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aeroembed.git
cd aeroembed
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Repository Structure

```
aeroembed/
├── src/aeroembed/              # Core library
│   ├── generators/             # TabSyn model implementation
│   │   └── tabsyn/            # VAE + Diffusion architecture
│   └── preprocessing/          # Data preprocessing utilities
│
├── scripts/                    # Analysis and visualization scripts
│   ├── embedding/             # Model training and embedding extraction
│   ├── analysis/              # Operational analysis tools
│   │   ├── operational_patterns/
│   │   ├── clustering/
│   │   └── temporal_analysis/
│   ├── visualization/         # Embedding space visualization
│   └── generation/            # Synthetic data generation
│
└── examples/                  # Example notebooks and data
```

## Quick Start

### 1. Train TabSyn and Extract Embeddings

```bash
# Train model and extract embeddings in one step
python scripts/embedding/train_and_extract_embeddings.py \
    --train_path data/flights_train.csv \
    --model_dir models/flight_model \
    --embeddings_output embeddings/flight_embeddings.npz \
    --vae_epochs 200 \
    --diffusion_epochs 1000
```

### 2. Extract Embeddings from Existing Model

```bash
# Extract embeddings only (if model already trained)
python scripts/embedding/extract_embeddings_from_model.py \
    --model_path models/flight_model/tabsyn_model.pkl \
    --data_path data/flights.csv \
    --output_path embeddings/flight_embeddings.npz
```

### 3. Analyze Operational Patterns

```bash
# Analyze airport networks
python scripts/analysis/operational_patterns/analyze_airport_network.py \
    --embeddings embeddings/flight_embeddings.npz \
    --data data/flights.csv \
    --output-dir results/airport_network

# Analyze carrier operations
python scripts/analysis/operational_patterns/analyze_carrier_operations.py \
    --embeddings embeddings/flight_embeddings.npz \
    --data data/flights.csv \
    --output-dir results/carrier_analysis
```

### 4. Detect Anomalies and Clusters

```bash
# Detect operational anomalies
python scripts/analysis/clustering/detect_anomalies.py \
    --embeddings embeddings/flight_embeddings.npz \
    --data data/flights.csv \
    --output-dir results/anomalies \
    --contamination 0.05

# Identify operational clusters
python scripts/analysis/clustering/detect_operational_clusters.py \
    --embeddings embeddings/flight_embeddings.npz \
    --data data/flights.csv \
    --output-dir results/clusters
```

## Data Format

The input flight data should be a CSV file with the following columns:

**Required columns:**
- `IATA_CARRIER_CODE`: Airline identifier
- `DEPARTURE_IATA_AIRPORT_CODE`: Departure airport code
- `ARRIVAL_IATA_AIRPORT_CODE`: Arrival airport code
- `AIRCRAFT_TYPE_IATA`: Aircraft type code
- `SCHEDULED_DEPARTURE_UTC`: Scheduled departure time
- `DEPARTURE_DELAY_MIN`: Departure delay in minutes

**Optional columns:**
- `TURNAROUND_MIN`: Turnaround time in minutes
- `SCHEDULED_DURATION_MIN`: Scheduled flight duration
- `ACTUAL_DURATION_MIN`: Actual flight duration

## Analysis Scripts Documentation

### Operational Pattern Analysis

#### Airport Network Analysis
```bash
python scripts/analysis/operational_patterns/analyze_airport_network.py --help
```
Creates a network graph of airports based on embedding similarity, revealing operational relationships beyond geographic proximity.

#### Carrier Operations Analysis
```bash
python scripts/analysis/operational_patterns/analyze_carrier_operations.py --help
```
Compares operational signatures across different carriers, visualizing their distinct patterns in embedding space.

#### Route Signature Analysis
```bash
python scripts/analysis/operational_patterns/analyze_route_signatures.py --help
```
Examines route-specific operational characteristics and their manifestation in the embedding space.

#### Seasonal Pattern Analysis
```bash
python scripts/analysis/operational_patterns/analyze_seasonal_patterns.py --help
```
Identifies seasonal variations in flight operations through temporal embedding analysis.

### Clustering and Anomaly Detection

#### Operational Clustering
```bash
python scripts/analysis/clustering/detect_operational_clusters.py --help
```
Uses HDBSCAN to identify natural groupings in flight operations, revealing common operational modes.

#### Anomaly Detection
```bash
python scripts/analysis/clustering/detect_anomalies.py --help
```
Compares Isolation Forest and HDBSCAN approaches for identifying unusual flight operations.

### Temporal Analysis

#### Delay Pattern Analysis
```bash
python scripts/analysis/temporal_analysis/analyze_delay_patterns.py --help
```
Explores how departure delays manifest in the embedding space across different operational contexts.

#### Turnaround Pattern Analysis
```bash
python scripts/analysis/temporal_analysis/analyze_turnaround_patterns.py --help
```
Analyzes aircraft turnaround time patterns and their relationship with operational performance.

### Visualization Tools

#### Embedding Space Navigation
```bash
python scripts/visualization/navigate_embedding_space.py --help
```
Creates directional maps showing how key operational features vary across the embedding space.

#### PCA Component Interpretation
```bash
python scripts/visualization/interpret_pca_components.py --help
```
Identifies and interprets the principal components of variation in the embedding space.
