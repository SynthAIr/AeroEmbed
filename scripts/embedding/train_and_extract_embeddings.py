#!/usr/bin/env python
"""
Train TabSyn model and extract embeddings for flight operational data analysis.

This script provides a unified interface for:
1. Training a TabSyn model on flight data
2. Extracting embeddings from the trained model
3. Saving embeddings for downstream analysis

Usage:
    # Train and extract embeddings
    python train_and_extract_embeddings.py \
        --train_path data/flights_train.csv \
        --model_dir models/flight_model \
        --embeddings_output embeddings/flight_embeddings.npz
    
    # Extract embeddings from existing model
    python train_and_extract_embeddings.py \
        --model_dir models/flight_model \
        --data_path data/flights_analysis.csv \
        --embeddings_output embeddings/flight_embeddings.npz \
        --extract_only
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict

from aeroembed.generators import TabSyn
from aeroembed.preprocessing import preprocess_flight_data

def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names=None):
    """
    Create mappings between column indices and names.
    
    Args:
        data_df: DataFrame containing the data
        num_col_idx: Indices of numerical columns
        cat_col_idx: Indices of categorical columns
        target_col_idx: Indices of target columns
        column_names: Optional list of column names
        
    Returns:
        tuple: Mappings between indices and names
    """
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    
    idx_mapping = {}
    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def split_and_save(df, split, save_dir, num_idx, cat_idx, tgt_idx):
    """
    Split data into features and target, and save as numpy arrays.
    
    Args:
        df: DataFrame to split
        split: Split name ('train' or 'test')
        save_dir: Directory to save files
        num_idx: Indices of numerical columns
        cat_idx: Indices of categorical columns
        tgt_idx: Indices of target columns
    """
    X_num = df.iloc[:, num_idx].astype(np.float32).to_numpy()
    X_cat = df.iloc[:, cat_idx].astype(str).to_numpy()
    y = df.iloc[:, tgt_idx].astype(np.float32).to_numpy().reshape(-1, 1)

    np.save(save_dir / f"X_num_{split}.npy", X_num)
    np.save(save_dir / f"X_cat_{split}.npy", X_cat)
    np.save(save_dir / f"y_{split}.npy", y)


def process_dataset(train_path, model_dir):
    """
    Preprocess flight data for TabSyn and save in the model directory.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        model_dir: Directory to save model and processed data
        
    Returns:
        str: Path to the processed dataset directory
    """
    # Create data directory within model_dir
    data_dir = Path(model_dir) / "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Default categorical columns for flight data
    cat_columns = [
        "IATA_CARRIER_CODE",
        "DEPARTURE_IATA_AIRPORT_CODE",
        "ARRIVAL_IATA_AIRPORT_CODE",
        "AIRCRAFT_TYPE_IATA",
    ]
    
    # Default target column for flight data
    target_column = "DEPARTURE_DELAY_MIN"
    
    # Load and preprocess data
    raw_train = pd.read_csv(train_path)
    
    train_df = preprocess_flight_data(raw_train)

    
    
    # Get column indices
    column_order = train_df.columns.tolist()
    
    num_idx = [i for i, c in enumerate(column_order)
               if c not in cat_columns + [target_column]]
    cat_idx = [column_order.index(c) for c in cat_columns]
    tgt_idx = [column_order.index(target_column)]
    
    # Save as numpy arrays
    split_and_save(train_df, "train", data_dir, num_idx, cat_idx, tgt_idx)

    
    # Create and save info.json
    idx_map, inv_map, idx_name_map = get_column_name_mapping(
        train_df, num_idx, cat_idx, tgt_idx, column_order)
    
    # Get dataset name from model directory
    dataset_name = os.path.basename(os.path.normpath(model_dir))
    
    info = {
        "name": dataset_name,
        "task_type": "regression",  # Flight delay is regression
        "header": "infer",
        "column_names": column_order,
        "num_col_idx": num_idx,
        "cat_col_idx": cat_idx,
        "target_col_idx": tgt_idx,
        "file_type": "csv",
        # Raw CSVs for reference
        "data_path": str(data_dir / "train.csv"),
        "test_path": str(data_dir / "test.csv"),
        # Helper mappings for TabSyn
        "idx_mapping": idx_map,
        "inverse_idx_mapping": inv_map,
        "idx_name_mapping": idx_name_map,
        "train_size": len(train_df),
    }
    
    with open(data_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    
    print(f"✅ Flight dataset processed and saved to {data_dir}")
    return str(data_dir)


def train_model(
    train_path: str,
    model_dir: str,
    vae_epochs: int = 200,
    diffusion_epochs: int = 1000,
    embedding_dim: int = 4,
    batch_size: int = 8192,
    device: str = "cuda",
    verbose: bool = True
) -> TabSyn:
    """
    Train a TabSyn model on flight data.
    
    Args:
        train_path: Path to training data CSV
        model_dir: Directory to save the model
        vae_epochs: Number of VAE training epochs
        diffusion_epochs: Number of diffusion training epochs
        embedding_dim: Dimension of the embedding space
        batch_size: Batch size for training
        device: Device to use for training
        verbose: Whether to print verbose output
        
    Returns:
        Trained TabSyn model
    """
    print(f"Training TabSyn model on {train_path}...")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize TabSyn model
    tabsyn = TabSyn(
        embedding_dim=embedding_dim,
        vae_epochs=vae_epochs,
        diffusion_epochs=diffusion_epochs,
        batch_size=batch_size,
        device=device,
        verbose=verbose
    )
    
    # Train the model
    data_dir = process_dataset(train_path, model_dir)
    tabsyn.fit(data_dir, task_type="regression")
    
    # Save the model
    model_path = os.path.join(model_dir, "tabsyn_model.pkl")
    tabsyn.save(model_path)
    
    print(f"Model trained and saved to {model_path}")
    return tabsyn


def extract_embeddings(
    model_path: str,
    data_path: str,
    batch_size: int = 1024,
    device: str = "cuda"
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Extract embeddings from a trained TabSyn model.
    
    Args:
        model_path: Path to the saved TabSyn model
        data_path: Path to the data to encode
        batch_size: Batch size for encoding
        device: Device to use for encoding
        
    Returns:
        Tuple of (embeddings, original_data, metadata)
    """
    print(f"Loading model from {model_path}...")
    
    # Load the model
    tabsyn = TabSyn.load(model_path)
    tabsyn.device = device
    
    # Load and preprocess the data
    print(f"Loading data from {data_path}...")
    raw_data = pd.read_csv(data_path)
    processed_data = preprocess_flight_data(raw_data)
    
    # Extract embeddings using the encoder
    print("Extracting embeddings...")
    
    # Get the encoder and decoder from the model
    encoder = tabsyn.encoder.to(device)
    encoder.eval()
    
    # Prepare data for encoding
    # This follows the same preprocessing as training
    from aeroembed.generators.tabsyn.data_utils import preprocess, TabularDataset
    
    # Get dataset info from model directory
    model_dir = os.path.dirname(model_path)
    info_path = os.path.join(model_dir, "data", "info.json")
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Preprocess data
    X_num, X_cat, categories, d_numerical = preprocess(
        os.path.join(model_dir, "data"),
        task_type=info['task_type']
    )
    
    # Create dataset
    X_num_tensor = torch.tensor(X_num[0]).float()  # Use training data
    X_cat_tensor = torch.tensor(X_cat[0])
    dataset = TabularDataset(X_num_tensor, X_cat_tensor)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Extract embeddings
    all_embeddings = []
    
    with torch.no_grad():
        for batch_num, batch_cat in dataloader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            
            # Get embeddings from encoder
            z = encoder(batch_num, batch_cat)
            
            # Extract mean (μ) from the encoder output
            # The encoder returns the latent representation
            embeddings = z[:, 1:, :]  # Remove CLS token
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Clear GPU memory
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Create metadata dictionary
    metadata = {
        'carrier': processed_data['IATA_CARRIER_CODE'].values,
        'departure_airport': processed_data['DEPARTURE_IATA_AIRPORT_CODE'].values,
        'arrival_airport': processed_data['ARRIVAL_IATA_AIRPORT_CODE'].values,
        'aircraft_type': processed_data['AIRCRAFT_TYPE_IATA'].values,
        'delay': processed_data['DEPARTURE_DELAY_MIN'].values,
        'turnaround': processed_data.get('TURNAROUND_MIN', np.array([])).values
    }
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings, processed_data, metadata


def save_embeddings(
    embeddings: np.ndarray,
    data: pd.DataFrame,
    metadata: Dict[str, np.ndarray],
    output_path: str
):
    """
    Save embeddings and associated metadata to disk.
    
    Args:
        embeddings: The extracted embeddings
        data: Original dataframe
        metadata: Dictionary of metadata arrays
        output_path: Path to save the embeddings
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as compressed numpy archive
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        **metadata
    )
    
    # Also save the full dataframe as CSV for reference
    csv_path = output_path.replace('.npz', '_data.csv')
    data.to_csv(csv_path, index=False)
    
    # Save embedding info
    info = {
        'embedding_shape': embeddings.shape,
        'n_samples': len(embeddings),
        'metadata_keys': list(metadata.keys()),
        'model_type': 'TabSyn',
        'extraction_date': pd.Timestamp.now().isoformat()
    }
    
    info_path = output_path.replace('.npz', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"Embeddings saved to {output_path}")
    print(f"Data saved to {csv_path}")
    print(f"Info saved to {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train TabSyn and extract embeddings for flight data analysis"
    )
    
    # Model paths
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory for the TabSyn model"
    )
    
    # Data paths
    parser.add_argument(
        "--train_path",
        type=str,
        help="Path to training data CSV (required for training)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to data for embedding extraction"
    )
    
    # Output
    parser.add_argument(
        "--embeddings_output",
        type=str,
        required=True,
        help="Path to save extracted embeddings (.npz file)"
    )
    
    # Mode
    parser.add_argument(
        "--extract_only",
        action="store_true",
        help="Only extract embeddings from existing model"
    )
    
    # Training parameters
    parser.add_argument(
        "--vae_epochs",
        type=int,
        default=200,
        help="Number of VAE training epochs"
    )
    parser.add_argument(
        "--diffusion_epochs",
        type=int,
        default=1000,
        help="Number of diffusion training epochs"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=4,
        help="Dimension of the embedding space"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.extract_only and not args.train_path:
        parser.error("--train_path is required when training a new model")
    
    if args.extract_only and not args.data_path:
        args.data_path = args.train_path  # Use training data if not specified
    
    # Train model if needed
    if not args.extract_only:
        tabsyn = train_model(
            train_path=args.train_path,
            model_dir=args.model_dir,
            vae_epochs=args.vae_epochs,
            diffusion_epochs=args.diffusion_epochs,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            device=args.device,
            verbose=args.verbose
        )
        # Use training data for embedding extraction if not specified
        if not args.data_path:
            args.data_path = args.train_path
    
    # Extract embeddings
    model_path = os.path.join(args.model_dir, "tabsyn_model.pkl")
    embeddings, data, metadata = extract_embeddings(
        model_path=model_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Save embeddings
    save_embeddings(
        embeddings=embeddings,
        data=data,
        metadata=metadata,
        output_path=args.embeddings_output
    )
    
    print("Process completed successfully!")


if __name__ == "__main__":
    main()