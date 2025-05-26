#!/usr/bin/env python
"""
Extract embeddings from tabular data using a trained TabSyn model.

Usage:
    python extract_embeddings.py --data_path data/new_data.csv --model_dir models/tabsyn --output_path embeddings/new_data_embeddings.npy
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder

from aeroembed.preprocessing import preprocess_flight_data
from aeroembed.generators.tabsyn.data_utils import TabularDataset
from aeroembed.generators.tabsyn.vae.model import Encoder_model


def load_encoder(model_dir, device='cuda'):
    """
    Load the trained encoder from the model directory.
    
    Args:
        model_dir: Path to the model directory
        device: Device to load the model on
        
    Returns:
        tuple: (encoder, info, categories, d_numerical)
    """
    # Load model info
    info_path = os.path.join(model_dir, 'data', 'info.json')
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Load training parameters
    params_path = os.path.join(model_dir, 'training_params.json')
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Load config to get categories and d_numerical
    config_path = os.path.join(model_dir, 'ckpt', 'config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    categories = config['categories']
    d_numerical = config['d_numerical']
    
    # Initialize encoder
    encoder = Encoder_model(
        params.get("vae_layers", 2),
        d_numerical,
        categories,
        params.get("embedding_dim", 4),
        n_head=params.get("n_head", 1),
        factor=params.get("vae_factor", 32)
    ).to(device)
    
    # Load encoder weights
    encoder_path = os.path.join(model_dir, 'ckpt', 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    
    return encoder, info, categories, d_numerical


def load_training_data_info(model_dir):
    """
    Load information about the training data format.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        dict: Information about training data format
    """
    dataset_dir = os.path.join(model_dir, 'data')
    
    # Load training data to understand format
    X_num_train = np.load(os.path.join(dataset_dir, 'X_num_train.npy'), allow_pickle=True)
    X_cat_train = np.load(os.path.join(dataset_dir, 'X_cat_train.npy'), allow_pickle=True)
    y_train     = np.load(os.path.join(dataset_dir, 'y_train.npy'),     allow_pickle=True)
    
    # Create categorical mappings
    cat_mappings = []
    for col_idx in range(X_cat_train.shape[1]):
        unique_values = np.unique(X_cat_train[:, col_idx])
        cat_mappings.append(unique_values)
    
    return {
        'num_mean': X_num_train.mean(axis=0),
        'num_std': X_num_train.std(axis=0),
        'cat_mappings': cat_mappings,
        'X_num_train': X_num_train,
        'X_cat_train': X_cat_train,
        'y_train': y_train
    }


def preprocess_data(data_path, info, train_info):
    """
    Preprocess input CSV to match TabSyn training format.
    """
    df = pd.read_csv(data_path)

    # Detect flight data and apply flight-specific preprocessing
    flight_cols = [
        'IATA_CARRIER_CODE', 'DEPARTURE_IATA_AIRPORT_CODE', 'ARRIVAL_IATA_AIRPORT_CODE'
    ]
    if all(col in df.columns for col in flight_cols):
        df = preprocess_flight_data(df)

    # Column indices from info.json
    num_idx    = info['num_col_idx']
    cat_idx    = info['cat_col_idx']
    tgt_idx    = info['target_col_idx']
    task_type  = info['task_type']

    # Build numeric matrix (include target first for regression)
    num_parts = []
    if task_type == 'regression':
        y = df.iloc[:, tgt_idx].astype(np.float32).to_numpy().reshape(-1,1)
        num_parts.append(y)
    if num_idx:
        Xn = df.iloc[:, num_idx].astype(np.float32).to_numpy()
        num_parts.append(Xn)
    X_num = np.concatenate(num_parts, axis=1) if num_parts else np.empty((len(df),0), dtype=np.float32)

    # Build raw categorical matrix (include target for classification)
    cat_parts = []
    if task_type != 'regression':
        y_cat = df.iloc[:, tgt_idx].astype(str).to_numpy().reshape(-1,1)
        cat_parts.append(y_cat)
    if cat_idx:
        Xc = df.iloc[:, cat_idx].astype(str).to_numpy()
        cat_parts.append(Xc)
    X_cat_raw = np.concatenate(cat_parts, axis=1) if cat_parts else np.empty((len(df),0), dtype=object)

    # Quantile-transform numeric features using training distribution
    if X_num.shape[1] > 0:
        if task_type == 'regression':
            full_train = np.concatenate([
                train_info['y_train'],
                train_info['X_num_train']
            ], axis=1)
        else:
            full_train = train_info['X_num_train']

        qt = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(full_train.shape[0]//30, 1000), 10),
            subsample=int(1e9),
            random_state=0
        )
        qt.fit(full_train)
        X_num = qt.transform(X_num)

    # Ordinally encode categorical features
    if X_cat_raw.shape[1] > 0:
        unknown_val = np.iinfo(np.int64).max - 3
        oe = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=unknown_val,
            dtype=np.int64
        )
        oe.fit(train_info['X_cat_train'])
        X_cat = oe.transform(X_cat_raw).astype(np.int64)
    else:
        X_cat = np.empty((len(df),0), dtype=np.int64)

    return X_num, X_cat



def extract_embeddings(
    data_path,
    model_dir,
    output_path=None,
    batch_size=1024,
    device='cuda',
    include_cls_token=False
):
    """
    Extract embeddings from tabular data using a trained TabSyn model.
    
    Args:
        data_path: Path to the input data CSV file
        model_dir: Path to the trained model directory
        output_path: Path to save the embeddings (optional)
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
        include_cls_token: Whether to include the CLS token in the output
        
    Returns:
        np.ndarray: Extracted embeddings
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Loading encoder from {model_dir}...")
    encoder, info, categories, d_numerical = load_encoder(model_dir, device)
    
    print(f"Loading training data information...")
    train_info = load_training_data_info(model_dir)
    
    print(f"Preprocessing data from {data_path}...")
    X_num, X_cat = preprocess_data(data_path, info, train_info)
    
    # Convert to torch tensors
    X_num = torch.tensor(X_num, dtype=torch.float32)
    X_cat = torch.tensor(X_cat, dtype=torch.long)
    
    print(f"Extracting embeddings for {len(X_num)} samples...")
    print(f"Numerical features shape: {X_num.shape}")
    print(f"Categorical features shape: {X_cat.shape}")
    
    # Extract embeddings in batches
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(X_num), batch_size):
            end_idx = min(i + batch_size, len(X_num))
            
            # Get batch
            batch_num = X_num[i:end_idx].to(device)
            batch_cat = X_cat[i:end_idx].to(device)
            
            # Get embeddings
            batch_z = encoder(batch_num, batch_cat).detach().cpu()
            all_embeddings.append(batch_z)
            
            # Free GPU memory
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # Progress update
            if (i + batch_size) % (batch_size * 10) == 0 or end_idx == len(X_num):
                print(f"Processed {end_idx}/{len(X_num)} samples...")
    
    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    
    # Remove CLS token if requested (default behavior to match training)
    if not include_cls_token:
        embeddings = embeddings[:, 1:, :]
        print(f"Removed CLS token. Final embedding shape: {embeddings.shape}")
    else:
        print(f"Keeping CLS token. Final embedding shape: {embeddings.shape}")
    
    # Save embeddings if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, embeddings)
        print(f"Embeddings saved to {output_path}")
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from tabular data using a trained TabSyn model")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input data CSV file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the trained TabSyn model directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the extracted embeddings (NPY file)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation"
    )
    parser.add_argument(
        "--include_cls_token",
        action="store_true",
        help="Include the CLS token in the output embeddings"
    )
    
    args = parser.parse_args()
    
    # If no output path specified, create one based on input
    if args.output_path is None:
        input_name = Path(args.data_path).stem
        args.output_path = f"embeddings/{input_name}_embeddings.npy"
    
    # Extract embeddings
    embeddings = extract_embeddings(
        data_path=args.data_path,
        model_dir=args.model_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        device=args.device,
        include_cls_token=args.include_cls_token
    )
    
    print(f"\nExtraction complete!")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"- Samples: {embeddings.shape[0]}")
    print(f"- Tokens per sample: {embeddings.shape[1]}")
    print(f"- Embedding dimension: {embeddings.shape[2]}")


if __name__ == "__main__":
    main()


# Example usage:
# python scripts/embedding/extract_embeddings.py --data_path data/new_flights.csv --model_dir models/tabsyn --output_path embeddings/new_flights_embeddings.npy
#
# Extract embeddings from test data:
# python scripts/embedding/extract_embeddings.py --data_path ../syntabair/data/real/test.csv --model_dir models/tabsyn
#
# Extract with CLS token included:
# python scripts/embedding/extract_embeddings.py --data_path data/flights.csv --model_dir models/tabsyn --include_cls_token