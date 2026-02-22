#!/usr/bin/env python

import pandas as pd
import torch
import ast
import re
from train_nmr import parse_1H_peaks, parse_13C_peaks, SMILESTokenizer

def validate_dataset(df):
    """Validate the dataset for potential issues."""
    print("Validating dataset...")
    print(f"Total rows: {len(df)}")
    print(f"Unique SMILES: {df['SMILES'].nunique()}")
    print(f"NMR types: {df['NMR_type'].unique()}")
    print()
    
    # Check for missing values
    print("Checking for missing values:")
    print(df.isnull().sum())
    print()
    
    # Check NMR_processed column
    print("Checking NMR_processed column:")
    empty_processed = df[df['NMR_processed'].isnull() | (df['NMR_processed'] == '') | (df['NMR_processed'] == '[]')]
    print(f"Empty/NULL NMR_processed entries: {len(empty_processed)}")
    print()
    
    # Check for NaN/Inf in chemical shifts
    print("Checking for NaN/Inf in chemical shifts:")
    nan_shifts = 0
    inf_shifts = 0
    
    for idx, row in df.iterrows():
        try:
            if row['NMR_type'] == '1H NMR':
                peaks = parse_1H_peaks(row['NMR_processed'])
                if torch.isnan(peaks).any():
                    print(f"Row {idx}: NaN in 1H peaks")
                    nan_shifts += 1
                if torch.isinf(peaks).any():
                    print(f"Row {idx}: Inf in 1H peaks")
                    inf_shifts += 1
            elif row['NMR_type'] == '13C NMR':
                peaks = parse_13C_peaks(row['NMR_processed'])
                if torch.isnan(peaks).any():
                    print(f"Row {idx}: NaN in 13C peaks")
                    nan_shifts += 1
                if torch.isinf(peaks).any():
                    print(f"Row {idx}: Inf in 13C peaks")
                    inf_shifts += 1
        except Exception as e:
            print(f"Row {idx}: Error parsing - {e}")
    
    print(f"Total NaN shifts: {nan_shifts}")
    print(f"Total Inf shifts: {inf_shifts}")
    print()
    
    # Check SMILES validity
    print("Checking SMILES validity:")
    tokenizer = SMILESTokenizer()
    invalid_smiles = 0
    
    for idx, row in df.iterrows():
        try:
            tokenizer.encode(row['SMILES'], update_vocab=False)
        except Exception as e:
            print(f"Row {idx}: Invalid SMILES - {row['SMILES']}")
            invalid_smiles += 1
    
    print(f"Invalid SMILES: {invalid_smiles}")
    print()
    
    # Check peak statistics
    print("Peak statistics:")
    h1_peak_counts = []
    c13_peak_counts = []
    
    for idx, row in df.iterrows():
        try:
            if row['NMR_type'] == '1H NMR':
                peaks = parse_1H_peaks(row['NMR_processed'])
                h1_peak_counts.append(peaks.size(0))
            elif row['NMR_type'] == '13C NMR':
                peaks = parse_13C_peaks(row['NMR_processed'])
                c13_peak_counts.append(peaks.size(0))
        except:
            pass
    
    if h1_peak_counts:
        print(f"1H NMR - Min peaks: {min(h1_peak_counts)}, Max peaks: {max(h1_peak_counts)}, Avg peaks: {sum(h1_peak_counts)/len(h1_peak_counts):.2f}")
    if c13_peak_counts:
        print(f"13C NMR - Min peaks: {min(c13_peak_counts)}, Max peaks: {max(c13_peak_counts)}, Avg peaks: {sum(c13_peak_counts)/len(c13_peak_counts):.2f}")
    print()
    
    # Check for extreme values
    print("Checking for extreme chemical shift values:")
    extreme_h1 = []
    extreme_c13 = []
    
    for idx, row in df.iterrows():
        try:
            if row['NMR_type'] == '1H NMR':
                peaks = parse_1H_peaks(row['NMR_processed'])
                if peaks.size(0) > 0:
                    shifts = peaks[:, 0]  # First column is shift
                    if shifts.min() < 0 or shifts.max() > 15:
                        print(f"Row {idx}: Extreme 1H shifts - min: {shifts.min():.2f}, max: {shifts.max():.2f}")
                        extreme_h1.append(idx)
            elif row['NMR_type'] == '13C NMR':
                peaks = parse_13C_peaks(row['NMR_processed'])
                if peaks.size(0) > 0:
                    shifts = peaks[:, 0]  # First column is shift
                    if shifts.min() < 0 or shifts.max() > 250:
                        print(f"Row {idx}: Extreme 13C shifts - min: {shifts.min():.2f}, max: {shifts.max():.2f}")
                        extreme_c13.append(idx)
        except:
            pass
    
    print(f"Extreme 1H shift entries: {len(extreme_h1)}")
    print(f"Extreme 13C shift entries: {len(extreme_c13)}")

if __name__ == "__main__":
    df_big = pd.read_parquet("/home/joosep/17296666/NMRexp_10to24_1_1004_sc_less_than_1.parquet")
    
    print("=" * 80)
    print("FULL DATASET VALIDATION")
    print("=" * 80)
    validate_dataset(df_big)
    
    print("\n" + "=" * 80)
    print("SUBSET (first 10000) VALIDATION")
    print("=" * 80)
    validate_dataset(df_big.head(10000))
