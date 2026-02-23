#!/usr/bin/env python
"""
Main training script for NMRTrans model.
This script orchestrates the training process using the modular components.
"""

import pandas as pd
import torch
from src.models.nmrtrans import train_nmrtrans_lightning
from configs.data_paths import DATASET_PATH

def main():
    """Main entry point for training."""
    # Load the dataset
    df_big = pd.read_parquet(DATASET_PATH)
    
    # Start training
    train_nmrtrans_lightning(df_big)

if __name__ == "__main__":
    main()
