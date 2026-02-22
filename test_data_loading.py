#!/usr/bin/env python
"""
Standalone tests for data loading functionality in train_nmr.py
These tests can be run inside the singularity container.
"""

import sys
import os
import tempfile
import json
import ast

# Add the parent directory to path to import train_nmr modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import the functions we need to test
from train_nmr import (
    parse_1H_peaks, 
    parse_13C_peaks, 
    NMRDataset, 
    SMILESTokenizer,
    nmr_collate_fn,
    get_dataloader,
    MULT_MAP
)


def test_parse_1H_peaks():
    """Test parsing of 1H NMR peak data"""
    print("Testing parse_1H_peaks...")
    
    # Test case 1: Single peak with all features
    peak_data = [('s', '["7.2", "7.3"]', '1H', '7.25', '7.26')]
    result = parse_1H_peaks(peak_data)
    assert result.shape == (1, 9), f"Expected shape (1, 9), got {result.shape}"
    assert result[0, 0] == 7.255, f"Expected shift 7.255, got {result[0, 0]}"
    assert result[0, 1] == 1.0, f"Expected integration 1.0, got {result[0, 1]}"
    assert result[0, 2] == 0, f"Expected multiplicity 0 (singlet), got {result[0, 2]}"
    print("  ✓ Single peak parsing works")
    
    # Test case 2: Multiple peaks
    peak_data = [
        ('s', '["7.2", "7.3"]', '1H', '7.25', '7.26'),
        ('d', '["1.2", "1.3"]', '2H', '1.25', '1.26'),
    ]
    result = parse_1H_peaks(peak_data)
    assert result.shape == (2, 9), f"Expected shape (2, 9), got {result.shape}"
    print("  ✓ Multiple peaks parsing works")
    
    # Test case 3: Peak with multiple J-couplings (should be sorted)
    peak_data = [('t', '["7.2", "7.3", "7.1"]', '3H', '7.25', '7.26')]
    result = parse_1H_peaks(peak_data)
    # J-couplings should be sorted in descending order
    j_couplings = result[0, 3:9]
    assert torch.all(j_couplings[:-1] >= j_couplings[1:]), "J-couplings should be sorted in descending order"
    print("  ✓ J-coupling sorting works")
    
    # Test case 4: Empty data
    result = parse_1H_peaks([])
    assert result.shape == (0, 9), f"Expected shape (0, 9), got {result.shape}"
    print("  ✓ Empty data handling works")
    
    # Test case 5: String input (as stored in parquet)
    peak_data_str = "[('s', '[\"7.2\", \"7.3\"]', '1H', '7.25', '7.26')]"
    result = parse_1H_peaks(peak_data_str)
    assert result.shape == (1, 9), f"Expected shape (1, 9), got {result.shape}"
    print("  ✓ String input parsing works")
    
    # Test case 6: Peak with no J-couplings (should be padded with zeros)
    peak_data = [('s', '[]', '1H', '7.25', '7.26')]
    result = parse_1H_peaks(peak_data)
    j_couplings = result[0, 3:9]
    assert torch.all(j_couplings == 0.0), "J-couplings should be padded with zeros"
    print("  ✓ Zero-padding for missing J-couplings works")
    
    print("✓ All parse_1H_peaks tests passed!\n")


def test_parse_13C_peaks():
    """Test parsing of 13C NMR peak data"""
    print("Testing parse_13C_peaks...")
    
    # Test case 1: Single peak with shift
    peak_data = [(125.5, 's', '1C')]
    result = parse_13C_peaks(peak_data)
    assert result.shape == (1, 1), f"Expected shape (1, 1), got {result.shape}"
    assert result[0, 0] == 125.5, f"Expected shift 125.5, got {result[0, 0]}"
    print("  ✓ Single peak parsing works")
    
    # Test case 2: Multiple peaks
    peak_data = [(125.5, 's', '1C'), (130.2, 's', '1C')]
    result = parse_13C_peaks(peak_data)
    assert result.shape == (2, 1), f"Expected shape (2, 1), got {result.shape}"
    print("  ✓ Multiple peaks parsing works")
    
    # Test case 3: Peak with range (should take midpoint)
    peak_data = [([125.0, 126.0], 's', '1C')]
    result = parse_13C_peaks(peak_data)
    assert result[0, 0] == 125.5, f"Expected midpoint 125.5, got {result[0, 0]}"
    print("  ✓ Range midpoint calculation works")
    
    # Test case 4: Empty data
    result = parse_13C_peaks([])
    assert result.shape == (0, 1), f"Expected shape (0, 1), got {result.shape}"
    print("  ✓ Empty data handling works")
    
    # Test case 5: String input
    peak_data_str = "[(125.5, 's', '1C')]"
    result = parse_13C_peaks(peak_data_str)
    assert result.shape == (1, 1), f"Expected shape (1, 1), got {result.shape}"
    print("  ✓ String input parsing works")
    
    print("✓ All parse_13C_peaks tests passed!\n")


def test_smiles_tokenizer():
    """Test SMILES tokenizer functionality"""
    print("Testing SMILESTokenizer...")
    
    tokenizer = SMILESTokenizer()
    
    # Test case 1: Basic SMILES tokenization
    smiles = "CCO"
    tokens = tokenizer.tokenize(smiles)
    assert tokens == ['C', 'C', 'O'], f"Expected ['C', 'C', 'O'], got {tokens}"
    print("  ✓ Basic tokenization works")
    
    # Test case 2: SMILES with special characters
    smiles = "C[C@H](O)C"
    tokens = tokenizer.tokenize(smiles)
    assert '[C@H]' in tokens, "Should handle stereochemistry"
    print("  ✓ Special character handling works")
    
    # Test case 3: Encoding and decoding
    smiles = "CCO"
    encoded = tokenizer.encode(smiles, update_vocab=True)
    decoded = tokenizer.decode(encoded)
    assert decoded == smiles, f"Expected '{smiles}', got '{decoded}'"
    print("  ✓ Encoding/decoding round-trip works")
    
    # Test case 4: Special tokens
    assert tokenizer.vocab2id['[PAD]'] == 0
    assert tokenizer.vocab2id['[BOS]'] == 1
    assert tokenizer.vocab2id['[EOS]'] == 2
    assert tokenizer.vocab2id['[UNK]'] == 3
    print("  ✓ Special tokens are correctly initialized")
    
    # Test case 5: Unknown token handling
    smiles = "CCX"  # X is not in the pattern
    encoded = tokenizer.encode(smiles, update_vocab=True)
    # Should have UNK token
    assert tokenizer.vocab2id.get('X', None) is not None, "Should add new token to vocab"
    print("  ✓ Unknown token handling works")
    
    print("✓ All SMILESTokenizer tests passed!\n")


def test_nmr_dataset():
    """Test NMRDataset class"""
    print("Testing NMRDataset...")
    
    # Create test data
    data = {
        'SMILES': ['CCO', 'CCO', 'CCN'],
        'NMR_type': ['1H NMR', '13C NMR', '1H NMR'],
        'NMR_processed': [
            [('s', '["7.2", "7.3"]', '1H', '7.25', '7.26')],
            [(125.5, 's', '1C')],
            [('d', '["1.2", "1.3"]', '2H', '1.25', '1.26')]
        ]
    }
    df = pd.DataFrame(data)
    
    dataset = NMRDataset(df)
    
    # Test case 1: Dataset length
    assert len(dataset) == 2, f"Expected length 2 (unique SMILES), got {len(dataset)}"
    print("  ✓ Dataset length is correct")
    
    # Test case 2: Check that we can retrieve items
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert 'SMILES' in item, "Item should have SMILES"
        assert '1H' in item, "Item should have 1H data"
        assert '13C' in item, "Item should have 13C data"
        assert isinstance(item['SMILES'], str), "SMILES should be a string"
        assert isinstance(item['1H'], torch.Tensor), "1H should be a tensor"
        assert isinstance(item['13C'], torch.Tensor), "13C should be a tensor"
    print("  ✓ Items can be retrieved correctly")
    
    # Test case 3: Check that CCO has both modalities
    # Find the item with CCO SMILES
    cco_found = False
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item['SMILES'] == 'CCO':
            cco_found = True
            assert item['1H'].shape[0] > 0, "CCO should have 1H data"
            assert item['13C'].shape[0] > 0, "CCO should have 13C data"
            break
    assert cco_found, "Should find CCO in dataset"
    print("  ✓ CCO has both 1H and 13C data")
    
    # Test case 4: Check that CCN has only 1H data
    ccn_found = False
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item['SMILES'] == 'CCN':
            ccn_found = True
            assert item['1H'].shape[0] > 0, "CCN should have 1H data"
            assert item['13C'].shape[0] == 0, "CCN should not have 13C data"
            break
    assert ccn_found, "Should find CCN in dataset"
    print("  ✓ CCN has only 1H data")
    
    # Test case 5: Get item with only 13C data
    data2 = {
        'SMILES': ['CCO', 'CCO'],
        'NMR_type': ['13C NMR', '13C NMR'],
        'NMR_processed': [
            [(125.5, 's', '1C')],
            [(130.2, 's', '1C')]
        ]
    }
    df2 = pd.DataFrame(data2)
    dataset2 = NMRDataset(df2)
    
    item = dataset2[0]
    assert item['SMILES'] == 'CCO', f"Expected SMILES 'CCO', got {item['SMILES']}"
    assert item['1H'].shape[0] == 0, "Should not have 1H data"
    assert item['13C'].shape[0] > 0, "Should have 13C data"
    print("  ✓ Item with only 13C data retrieved correctly")
    
    print("✓ All NMRDataset tests passed!\n")


def test_collate_function():
    """Test the custom collate function"""
    print("Testing nmr_collate_fn...")
    
    # Create a simple batch
    batch = [
        {
            '1H': torch.tensor([[7.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            '13C': torch.tensor([[125.5]]),
            'SMILES': 'CCO'
        },
        {
            '1H': torch.tensor([[1.25, 2.0, 1.0, 7.2, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            '13C': torch.tensor([[130.2]]),
            'SMILES': 'CCN'
        }
    ]
    
    # Initialize tokenizer (needed for collate function)
    tokenizer = SMILESTokenizer()
    
    # Update tokenizer with test SMILES
    for item in batch:
        tokenizer.encode(item['SMILES'], update_vocab=True)
    
    # Collate the batch
    collated = nmr_collate_fn(batch)
    
    # Test case 1: Check shapes
    assert collated['1H'].shape == (2, 1, 9), f"Expected 1H shape (2, 1, 9), got {collated['1H'].shape}"
    assert collated['13C'].shape == (2, 1, 1), f"Expected 13C shape (2, 1, 1), got {collated['13C'].shape}"
    assert collated['SMILES_TOKENS'].shape[0] == 2, f"Expected batch size 2, got {collated['SMILES_TOKENS'].shape[0]}"
    print("  ✓ Collated batch has correct shapes")
    
    # Test case 2: Check masks
    assert collated['1H_mask'].shape == (2, 1), f"Expected 1H_mask shape (2, 1), got {collated['1H_mask'].shape}"
    assert collated['13C_mask'].shape == (2, 1), f"Expected 13C_mask shape (2, 1), got {collated['13C_mask'].shape}"
    assert torch.all(collated['1H_mask']), "All 1H peaks should be valid"
    assert torch.all(collated['13C_mask']), "All 13C peaks should be valid"
    print("  ✓ Masks are correctly generated")
    
    # Test case 3: Check padding values
    assert torch.all(collated['1H'][:, :, 0] != 0.0), "Shift values should not be zero"
    assert torch.all(collated['13C'][:, :, 0] != 0.0), "13C shift values should not be zero"
    print("  ✓ No accidental zero-padding of valid data")
    
    print("✓ All nmr_collate_fn tests passed!\n")


def test_dataloader():
    """Test the get_dataloader function"""
    print("Testing get_dataloader...")
    
    # Create test data
    data = {
        'SMILES': ['CCO', 'CCO', 'CCN', 'CCN', 'CCO'],
        'NMR_type': ['1H NMR', '13C NMR', '1H NMR', '13C NMR', '1H NMR'],
        'NMR_processed': [
            [('s', '["7.2", "7.3"]', '1H', '7.25', '7.26')],
            [(125.5, 's', '1C')],
            [('d', '["1.2", "1.3"]', '2H', '1.25', '1.26')],
            [(130.2, 's', '1C')],
            [('t', '["2.5", "2.6", "2.4"]', '3H', '2.55', '2.56')]
        ]
    }
    df = pd.DataFrame(data)
    
    # Get dataloader
    dataloader = get_dataloader(df)
    
    # Test case 1: Check dataloader properties
    assert dataloader.batch_size == 16, f"Expected batch_size 16, got {dataloader.batch_size}"
    # Note: shuffle is a parameter passed to DataLoader, not an attribute
    print("  ✓ Dataloader has correct properties")
    
    # Test case 2: Iterate through batches
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        # Check batch structure
        assert '1H' in batch, "Batch should contain '1H'"
        assert '13C' in batch, "Batch should contain '13C'"
        assert 'SMILES_TOKENS' in batch, "Batch should contain 'SMILES_TOKENS'"
        assert '1H_mask' in batch, "Batch should contain '1H_mask'"
        assert '13C_mask' in batch, "Batch should contain '13C_mask'"
        
        # Check tensor shapes
        assert batch['1H'].dim() == 3, "1H should be 3D tensor (batch, peaks, features)"
        assert batch['13C'].dim() == 3, "13C should be 3D tensor (batch, peaks, features)"
        assert batch['SMILES_TOKENS'].dim() == 2, "SMILES_TOKENS should be 2D tensor (batch, seq_len)"
        
        # Check masks
        assert batch['1H_mask'].shape == batch['1H'][:, :, 0].shape, "1H_mask should match 1H shape"
        assert batch['13C_mask'].shape == batch['13C'][:, :, 0].shape, "13C_mask should match 13C shape"
        
        break  # Only test first batch
    
    assert batch_count > 0, "Should have at least one batch"
    print("  ✓ Dataloader iteration works correctly")
    
    print("✓ All get_dataloader tests passed!\n")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    # Test case 1: Empty DataFrame
    df_empty = pd.DataFrame({'SMILES': [], 'NMR_type': [], 'NMR_processed': []})
    dataset = NMRDataset(df_empty)
    assert len(dataset) == 0, "Empty dataset should have length 0"
    print("  ✓ Empty DataFrame handling works")
    
    # Test case 2: DataFrame with only one modality per SMILES
    data = {
        'SMILES': ['CCO', 'CCN'],
        'NMR_type': ['1H NMR', '13C NMR'],
        'NMR_processed': [
            [('s', '["7.2", "7.3"]', '1H', '7.25', '7.26')],
            [(125.5, 's', '1C')]
        ]
    }
    df = pd.DataFrame(data)
    dataset = NMRDataset(df)
    
    # Find items by SMILES (order may vary due to grouping)
    cco_item = None
    ccn_item = None
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item['SMILES'] == 'CCO':
            cco_item = item
        elif item['SMILES'] == 'CCN':
            ccn_item = item
    
    assert cco_item is not None, "Should find CCO item"
    assert ccn_item is not None, "Should find CCN item"
    
    assert cco_item['1H'].shape[0] > 0, "CCO should have 1H data"
    assert cco_item['13C'].shape[0] == 0, "CCO should not have 13C data"
    
    assert ccn_item['1H'].shape[0] == 0, "CCN should not have 1H data"
    assert ccn_item['13C'].shape[0] > 0, "CCN should have 13C data"
    print("  ✓ Single modality per SMILES works")
    
    # Test case 3: Parse with None values
    result = parse_1H_peaks(None)
    assert result.shape == (0, 9), "Should return empty tensor for None input"
    
    result = parse_13C_peaks(None)
    assert result.shape == (0, 1), "Should return empty tensor for None input"
    print("  ✓ None value handling works")
    
    # Test case 4: Parse with empty list
    result = parse_1H_peaks([])
    assert result.shape == (0, 9), "Should return empty tensor for empty list"
    
    result = parse_13C_peaks([])
    assert result.shape == (0, 1), "Should return empty tensor for empty list"
    print("  ✓ Empty list handling works")
    
    print("✓ All edge case tests passed!\n")


def test_integration():
    """Test integration of all components"""
    print("Testing integration...")
    
    # Create comprehensive test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = 100
    smiles_list = ['CCO', 'CCN', 'CCO', 'CCN', 'CCO'] * 20
    
    data = {
        'SMILES': smiles_list,
        'NMR_type': ['1H NMR' if i % 2 == 0 else '13C NMR' for i in range(num_samples)],
        'NMR_processed': []
    }
    
    for i in range(num_samples):
        if data['NMR_type'][i] == '1H NMR':
            # Generate random 1H peak data
            num_peaks = np.random.randint(1, 5)
            peaks = []
            for _ in range(num_peaks):
                mult = np.random.choice(list(MULT_MAP.keys()))
                j_couplings = [str(np.random.uniform(0, 20)) for _ in range(np.random.randint(0, 4))]
                integration = str(np.random.uniform(0.5, 5.0))
                shift1 = str(np.random.uniform(0, 10))
                shift2 = str(np.random.uniform(0, 10))
                # Format J-couplings as string list
                j_str = '[' + ', '.join([f'"{j}"' for j in j_couplings]) + ']'
                peaks.append((mult, j_str, integration, shift1, shift2))
            data['NMR_processed'].append(peaks)
        else:
            # Generate random 13C peak data
            num_peaks = np.random.randint(1, 10)
            peaks = []
            for _ in range(num_peaks):
                shift = np.random.uniform(0, 200)
                peaks.append((shift, 's', '1C'))
            data['NMR_processed'].append(peaks)
    
    df = pd.DataFrame(data)
    
    # Test full pipeline
    dataloader = get_dataloader(df)
    
    batch_count = 0
    total_samples = 0
    for batch in dataloader:
        batch_count += 1
        batch_size = batch['1H'].shape[0]
        total_samples += batch_size
        
        # Verify batch integrity
        assert batch['1H'].shape == (batch_size, batch['1H'].shape[1], 9)
        assert batch['13C'].shape == (batch_size, batch['13C'].shape[1], 1)
        assert batch['SMILES_TOKENS'].shape == (batch_size, batch['SMILES_TOKENS'].shape[1])
        
        # Verify masks
        assert batch['1H_mask'].shape == (batch_size, batch['1H'].shape[1])
        assert batch['13C_mask'].shape == (batch_size, batch['13C'].shape[1])
    
    assert batch_count > 0, "Should have processed at least one batch"
    assert total_samples == len(df['SMILES'].unique()), f"Should process all unique SMILES, expected {len(df['SMILES'].unique())}, got {total_samples}"
    
    print(f"  ✓ Full pipeline processed {total_samples} samples in {batch_count} batches")
    print("✓ Integration test passed!\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running Data Loading Tests for train_nmr.py")
    print("=" * 60 + "\n")
    
    try:
        test_parse_1H_peaks()
        test_parse_13C_peaks()
        test_smiles_tokenizer()
        test_nmr_dataset()
        test_collate_function()
        test_dataloader()
        test_edge_cases()
        test_integration()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
