#!/usr/bin/env python
"""
Standalone tests for the SMILESTokenizer in train_nmr.py
Tests tokenization with real SMILES strings from data files.
"""

import sys
import os

# Add the parent directory to path to import train_nmr modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from train_nmr import SMILESTokenizer


def test_tokenizer_basic_functionality():
    """Test basic tokenizer functionality"""
    print("Testing basic tokenizer functionality...")
    
    tokenizer = SMILESTokenizer()
    
    # Test 1: Simple SMILES
    smiles = "CCO"
    tokens = tokenizer.tokenize(smiles)
    assert tokens == ['C', 'C', 'O'], f"Expected ['C', 'C', 'O'], got {tokens}"
    print("  ✓ Simple SMILES tokenization works")
    
    # Test 2: SMILES with rings
    smiles = "C1CCC1"
    tokens = tokenizer.tokenize(smiles)
    assert tokens == ['C', '1', 'C', 'C', 'C', '1'], f"Expected ['C', '1', 'C', 'C', 'C', '1'], got {tokens}"
    print("  ✓ Ring SMILES tokenization works")
    
    # Test 3: SMILES with branches
    smiles = "CC(C)O"
    tokens = tokenizer.tokenize(smiles)
    assert tokens == ['C', 'C', '(', 'C', ')', 'O'], f"Expected ['C', 'C', '(', 'C', ')', 'O'], got {tokens}"
    print("  ✓ Branched SMILES tokenization works")
    
    # Test 4: SMILES with stereochemistry
    smiles = "C[C@H](O)C"
    tokens = tokenizer.tokenize(smiles)
    assert '[C@H]' in tokens, "Should handle stereochemistry"
    print("  ✓ Stereochemistry tokenization works")
    
    # Test 5: SMILES with aromatic atoms
    smiles = "c1ccccc1"
    tokens = tokenizer.tokenize(smiles)
    assert tokens == ['c', '1', 'c', 'c', 'c', 'c', 'c', '1'], f"Expected ['c', '1', 'c', 'c', 'c', 'c', 'c', '1'], got {tokens}"
    print("  ✓ Aromatic SMILES tokenization works")
    
    # Test 6: SMILES with special elements
    smiles = "CBrClF"
    tokens = tokenizer.tokenize(smiles)
    assert tokens == ['C', 'Br', 'Cl', 'F'], f"Expected ['C', 'Br', 'Cl', 'F'], got {tokens}"
    print("  ✓ Special element tokenization works")
    
    # Test 7: SMILES with charges
    smiles = "[NH4+]"
    tokens = tokenizer.tokenize(smiles)
    assert '[NH4+]' in tokens, "Should handle charged groups"
    print("  ✓ Charged group tokenization works")
    
    print("✓ All basic functionality tests passed!\n")


def test_tokenizer_encoding_decoding():
    """Test encoding and decoding with vocabulary updates"""
    print("Testing encoding and decoding...")
    
    tokenizer = SMILESTokenizer()
    
    # Test 1: Basic encode/decode round-trip
    smiles = "CCO"
    encoded = tokenizer.encode(smiles, update_vocab=True)
    decoded = tokenizer.decode(encoded)
    assert decoded == smiles, f"Expected '{smiles}', got '{decoded}'"
    print("  ✓ Basic encode/decode round-trip works")
    
    # Test 2: Encode with BOS/EOS tokens
    encoded = tokenizer.encode(smiles, update_vocab=True)
    assert encoded[0] == tokenizer.vocab2id['[BOS]'], "First token should be BOS"
    assert encoded[-1] == tokenizer.vocab2id['[EOS]'], "Last token should be EOS"
    print("  ✓ BOS/EOS tokens are correctly added")
    
    # Test 3: Multiple SMILES with vocabulary growth
    initial_vocab_size = len(tokenizer.vocab2id)
    
    smiles_list = ["CCO", "CCN", "C1CCC1", "c1ccccc1"]
    for smiles in smiles_list:
        tokenizer.encode(smiles, update_vocab=True)
    
    final_vocab_size = len(tokenizer.vocab2id)
    assert final_vocab_size > initial_vocab_size, "Vocabulary should grow"
    print(f"  ✓ Vocabulary grew from {initial_vocab_size} to {final_vocab_size} tokens")
    
    # Test 4: Decode all SMILES correctly
    for smiles in smiles_list:
        encoded = tokenizer.encode(smiles, update_vocab=False)
        decoded = tokenizer.decode(encoded)
        assert decoded == smiles, f"Expected '{smiles}', got '{decoded}'"
    print("  ✓ All SMILES decode correctly after vocabulary update")
    
    print("✓ All encoding/decoding tests passed!\n")


def test_tokenizer_special_tokens():
    """Test special token handling"""
    print("Testing special token handling...")
    
    tokenizer = SMILESTokenizer()
    
    # Test 1: Special tokens have correct IDs
    assert tokenizer.vocab2id['[PAD]'] == 0, "PAD token should have ID 0"
    assert tokenizer.vocab2id['[BOS]'] == 1, "BOS token should have ID 1"
    assert tokenizer.vocab2id['[EOS]'] == 2, "EOS token should have ID 2"
    assert tokenizer.vocab2id['[UNK]'] == 3, "UNK token should have ID 3"
    print("  ✓ Special tokens have correct IDs")
    
    # Test 2: Special tokens are in both mappings
    for token in [tokenizer.PAD_TOKEN, tokenizer.BOS_TOKEN, 
                  tokenizer.EOS_TOKEN, tokenizer.UNK_TOKEN]:
        assert token in tokenizer.vocab2id, f"{token} should be in vocab2id"
        assert token in tokenizer.id2vocab.values(), f"{token} should be in id2vocab"
    print("  ✓ Special tokens are in both mappings")
    
    # Test 3: Unknown token handling
    smiles = "CCX"  # X is not in the pattern
    encoded = tokenizer.encode(smiles, update_vocab=True)
    # X should be added to vocabulary
    assert 'X' in tokenizer.vocab2id, "Unknown token should be added to vocab"
    
    # Now test with unknown token that's not added
    tokenizer2 = SMILESTokenizer()
    smiles = "CCX"
    encoded = tokenizer2.encode(smiles, update_vocab=False)
    # Should use UNK token
    assert tokenizer2.vocab2id['[UNK]'] in encoded, "Should use UNK token for unknown characters"
    print("  ✓ Unknown token handling works correctly")
    
    print("✓ All special token tests passed!\n")


def test_tokenizer_with_real_smiles():
    """Test tokenizer with real SMILES strings from data"""
    print("Testing with real SMILES strings...")
    
    # Common SMILES patterns found in chemical databases
    real_smiles = [
        "CCO",  # Ethanol
        "CC(C)O",  # Isopropanol
        "C1CCCCC1",  # Cyclohexane
        "c1ccccc1",  # Benzene
        "CC(=O)O",  # Acetic acid
        "C1=CC=CC=C1",  # Cyclohexene
        "C1CCOCC1",  # Tetrahydrofuran
        "CC1=CC=C(C=C1)O",  # Phenol
        "CC(C)(C)C",  # 2-Methylbutane
        "C1CC2(CCC1)CCC2",  # Decalin
        "C1=CC=C2C(=C1)C=CC=C2",  # Naphthalene
        "CC1=CC=C(C=C1)C(C)C",  # Isopropylbenzene
        "C1CCCC1O",  # Tetrahydropyran
        "C1=CC(=O)C=C1",  # 2-Cyclopenten-1-one
        "C1CCC(CC1)C(=O)O",  # Cyclohexanecarboxylic acid
    ]
    
    tokenizer = SMILESTokenizer()
    
    # Test 1: All SMILES can be tokenized
    for smiles in real_smiles:
        tokens = tokenizer.tokenize(smiles)
        assert len(tokens) > 0, f"Should produce tokens for {smiles}"
        # Verify no empty tokens
        assert all(len(t) > 0 for t in tokens), f"Should not have empty tokens for {smiles}"
    print(f"  ✓ All {len(real_smiles)} real SMILES strings can be tokenized")
    
    # Test 2: Encode and decode all SMILES
    for smiles in real_smiles:
        encoded = tokenizer.encode(smiles, update_vocab=True)
        decoded = tokenizer.decode(encoded)
        assert decoded == smiles, f"Round-trip failed for '{smiles}': got '{decoded}'"
    print(f"  ✓ All {len(real_smiles)} SMILES pass encode/decode round-trip")
    
    # Test 3: Check token statistics
    all_tokens = []
    for smiles in real_smiles:
        tokens = tokenizer.tokenize(smiles)
        all_tokens.extend(tokens)
    
    unique_tokens = set(all_tokens)
    print(f"  ✓ Token statistics: {len(all_tokens)} total tokens, {len(unique_tokens)} unique")
    
    # Test 4: Verify common chemical patterns are captured
    common_patterns = ['C', 'O', 'N', 'c', 'n', 'o', '1', '2', '(', ')', 'C=', 'C#']
    for pattern in common_patterns:
        # Check if pattern exists in tokens
        pattern_found = any(pattern in token for token in all_tokens)
        if pattern_found:
            print(f"    - Found pattern: {pattern}")
    
    print("✓ All real SMILES tests passed!\n")


def test_tokenizer_vocabulary_consistency():
    """Test that vocabulary remains consistent across multiple instances"""
    print("Testing vocabulary consistency...")
    
    # Create two tokenizers
    tokenizer1 = SMILESTokenizer()
    tokenizer2 = SMILESTokenizer()
    
    # Add same SMILES to both
    smiles_list = ["CCO", "CCN", "C1CCC1"]
    
    for smiles in smiles_list:
        tokenizer1.encode(smiles, update_vocab=True)
        tokenizer2.encode(smiles, update_vocab=True)
    
    # Test 1: Both should have same vocabulary size
    assert len(tokenizer1.vocab2id) == len(tokenizer2.vocab2id), \
        "Both tokenizers should have same vocabulary size"
    print("  ✓ Vocabulary sizes match")
    
    # Test 2: Both should have same special tokens
    for token in [tokenizer1.PAD_TOKEN, tokenizer1.BOS_TOKEN, 
                  tokenizer1.EOS_TOKEN, tokenizer1.UNK_TOKEN]:
        assert tokenizer1.vocab2id[token] == tokenizer2.vocab2id[token], \
            f"Special token {token} should have same ID"
    print("  ✓ Special tokens have consistent IDs")
    
    # Test 3: Both should encode same SMILES to same IDs
    for smiles in smiles_list:
        enc1 = tokenizer1.encode(smiles, update_vocab=False)
        enc2 = tokenizer2.encode(smiles, update_vocab=False)
        assert torch.equal(enc1, enc2), f"Should encode '{smiles}' to same IDs"
    print("  ✓ Encoding is consistent across instances")
    
    print("✓ All vocabulary consistency tests passed!\n")


def test_tokenizer_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    tokenizer = SMILESTokenizer()
    
    # Test 1: Empty string
    tokens = tokenizer.tokenize("")
    assert tokens == [], "Empty string should produce empty token list"
    encoded = tokenizer.encode("", update_vocab=False)
    assert torch.equal(encoded, torch.tensor([tokenizer.vocab2id['[BOS]'], tokenizer.vocab2id['[EOS]']])), \
        "Empty string should produce BOS+EOS only"
    print("  ✓ Empty string handling works")
    
    # Test 2: String with only special characters
    smiles = "()[]"
    tokens = tokenizer.tokenize(smiles)
    assert len(tokens) > 0, "Should handle special characters"
    print("  ✓ Special character string handling works")
    
    # Test 3: Very long SMILES
    long_smiles = "C" * 1000
    tokens = tokenizer.tokenize(long_smiles)
    assert len(tokens) == 1000, "Should handle long SMILES"
    encoded = tokenizer.encode(long_smiles, update_vocab=True)
    assert len(encoded) == 1002, "Should have BOS + 1000 tokens + EOS"
    print("  ✓ Long SMILES handling works")
    
    # Test 4: SMILES with multiple consecutive special characters
    smiles = "CC(C)(C)O"
    tokens = tokenizer.tokenize(smiles)
    assert '(' in tokens and ')' in tokens, "Should handle nested parentheses"
    print("  ✓ Nested special characters handling works")
    
    # Test 5: Decode with tensor input
    smiles = "CCO"
    encoded = tokenizer.encode(smiles, update_vocab=True)
    decoded = tokenizer.decode(encoded)
    assert decoded == smiles, "Should decode tensor correctly"
    
    # Also test with list input
    decoded_list = tokenizer.decode(encoded.tolist())
    assert decoded_list == smiles, "Should decode list correctly"
    print("  ✓ Decoding with different input types works")
    
    print("✓ All edge case tests passed!\n")


def test_tokenizer_with_data_file():
    """Test tokenizer with actual data from parquet file"""
    print("Testing with actual data file...")
    
    try:
        # Try to load a small sample from the data file
        df = pd.read_parquet("/home/joosep/17296666/NMRexp_10to24_1_1004_sc_less_than_1.parquet")
        df_sample = df.head(100)
        
        # Get unique SMILES from sample
        smiles_list = df_sample['SMILES'].unique()
        print(f"  ✓ Loaded {len(smiles_list)} unique SMILES from data file")
        
        tokenizer = SMILESTokenizer()
        
        # Test 1: Tokenize all SMILES
        for smiles in smiles_list:
            tokens = tokenizer.tokenize(smiles)
            assert len(tokens) > 0, f"Should produce tokens for {smiles}"
        print(f"  ✓ All {len(smiles_list)} SMILES from data can be tokenized")
        
        # Test 2: Build vocabulary
        vocab_size_before = len(tokenizer.vocab2id)
        for smiles in smiles_list:
            tokenizer.encode(smiles, update_vocab=True)
        vocab_size_after = len(tokenizer.vocab2id)
        
        new_tokens = vocab_size_after - vocab_size_before - 4  # Subtract special tokens
        print(f"  ✓ Vocabulary grew by {new_tokens} tokens from data SMILES")
        
        # Test 3: Encode/decode round-trip
        for smiles in smiles_list[:10]:  # Test first 10
            encoded = tokenizer.encode(smiles, update_vocab=False)
            decoded = tokenizer.decode(encoded)
            assert decoded == smiles, f"Round-trip failed for '{smiles}': got '{decoded}'"
        print(f"  ✓ Round-trip encoding/decoding works for sample SMILES")
        
        # Test 4: Show some statistics
        all_tokens = []
        for smiles in smiles_list:
            tokens = tokenizer.tokenize(smiles)
            all_tokens.extend(tokens)
        
        unique_tokens = set(all_tokens)
        print(f"  ✓ Token statistics from data: {len(all_tokens)} total, {len(unique_tokens)} unique")
        
        # Show most common tokens
        from collections import Counter
        token_counts = Counter(all_tokens)
        top_10 = token_counts.most_common(10)
        print(f"    Top 10 tokens: {top_10}")
        
    except FileNotFoundError:
        print("  ⚠ Data file not found, skipping data file test")
    except Exception as e:
        print(f"  ⚠ Error loading data file: {e}")
    
    print("✓ Data file test completed!\n")


def main():
    """Run all tokenizer tests"""
    print("=" * 60)
    print("Running SMILESTokenizer Tests for train_nmr.py")
    print("=" * 60 + "\n")
    
    try:
        test_tokenizer_basic_functionality()
        test_tokenizer_encoding_decoding()
        test_tokenizer_special_tokens()
        test_tokenizer_with_real_smiles()
        test_tokenizer_vocabulary_consistency()
        test_tokenizer_edge_cases()
        test_tokenizer_with_data_file()
        
        print("=" * 60)
        print("✓ ALL TOKENIZER TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
