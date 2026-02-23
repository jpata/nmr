#!/usr/bin/env python
"""
Main training script for NMRTrans model.
This script orchestrates the training process using the modular components.
"""

import pandas as pd
import torch
from src.models.nmrtrans import train_nmrtrans_lightning

def main():
    """Main entry point for training."""
    # Load the dataset
    df_big = pd.read_parquet("/home/joosep/17296666/NMRexp_10to24_1_1004_sc_less_than_1.parquet")
    
    # Start training
    train_nmrtrans_lightning(df_big)

if __name__ == "__main__":
    main()
