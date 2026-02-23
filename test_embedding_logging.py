#!/usr/bin/env python

"""
Test script to verify embedding logging functionality.
This script tests the modified train_nmr.py to ensure embeddings are properly extracted and logged.
"""

import torch
import sys
sys.path.insert(0, '/home/joosep/nmr')

from train_nmr import SMILESTokenizer, NMRTrans, NMRLightningModule

def test_embedding_extraction():
    """Test that embeddings can be extracted from the model."""
    print("Testing embedding extraction...")
    
    # Create a simple tokenizer
    tokenizer = SMILESTokenizer()
    
    # Build a small vocabulary
    test_smiles = ["CCO", "CCC", "CCN"]
    for smiles in test_smiles:
        tokenizer.encode(smiles, update_vocab=True)
    
    VOCAB_SIZE = len(tokenizer.vocab2id)
    PAD_IDX = tokenizer.vocab2id[tokenizer.PAD_TOKEN]
    
    # Create model
    model = NMRTrans(vocab_size=VOCAB_SIZE, d_model=256, max_smiles_len=128)
    
    # Create dummy input
    batch_size = 2
    h1_x = torch.randn(batch_size, 5, 9)  # (batch, num_peaks, features)
    h1_mask = torch.ones(batch_size, 5, dtype=torch.bool)
    c13_x = torch.randn(batch_size, 3, 1)
    c13_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    
    # Create dummy SMILES tokens
    smiles_tokens = []
    for smiles in test_smiles[:batch_size]:
        encoded = tokenizer.encode(smiles, update_vocab=False)
        smiles_tokens.append(encoded)
    
    # Pad SMILES tokens
    from torch.nn.utils.rnn import pad_sequence
    smiles_padded = pad_sequence(
        smiles_tokens,
        batch_first=True,
        padding_value=PAD_IDX
    )
    
    # Create target (shifted right)
    tgt_input = smiles_padded[:, :-1]
    # Create a dummy mask for testing
    tgt_mask = torch.zeros(batch_size, tgt_input.size(1), tgt_input.size(1))
    
    # Test forward pass without embeddings
    print("  Testing forward pass without embeddings...")
    logits = model(h1_x, h1_mask, c13_x, c13_mask, tgt_input, tgt_mask)
    print(f"    Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, tgt_input.size(1), VOCAB_SIZE), "Logits shape mismatch"
    
    # Test forward pass with embeddings
    print("  Testing forward pass with embeddings...")
    logits, embeddings = model(h1_x, h1_mask, c13_x, c13_mask, tgt_input, tgt_mask, return_embeddings=True)
    print(f"    Logits shape: {logits.shape}")
    print(f"    Embeddings shape: {embeddings.shape}")
    assert logits.shape == (batch_size, tgt_input.size(1), VOCAB_SIZE), "Logits shape mismatch"
    assert embeddings.dim() == 3, "Embeddings should be 3D (batch, seq_len, d_model)"
    
    print("  ✓ Embedding extraction test passed!")
    return True

def test_lightning_module():
    """Test that the Lightning module can handle the new forward signature."""
    print("\nTesting Lightning module...")
    
    # Create a simple tokenizer
    tokenizer = SMILESTokenizer()
    test_smiles = ["CCO", "CCC", "CCN"]
    for smiles in test_smiles:
        tokenizer.encode(smiles, update_vocab=True)
    
    VOCAB_SIZE = len(tokenizer.vocab2id)
    PAD_IDX = tokenizer.vocab2id[tokenizer.PAD_TOKEN]
    
    # Create Lightning module
    lightning_module = NMRLightningModule(
        vocab_size=VOCAB_SIZE,
        pad_idx=PAD_IDX,
        d_model=256,
        max_smiles_len=128
    )
    lightning_module.tokenizer = tokenizer
    
    # Create dummy input
    batch_size = 2
    h1_x = torch.randn(batch_size, 5, 9)
    h1_mask = torch.ones(batch_size, 5, dtype=torch.bool)
    c13_x = torch.randn(batch_size, 3, 1)
    c13_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    
    # Create dummy SMILES tokens
    smiles_tokens = []
    for smiles in test_smiles[:batch_size]:
        encoded = tokenizer.encode(smiles, update_vocab=False)
        smiles_tokens.append(encoded)
    
    # Pad SMILES tokens
    from torch.nn.utils.rnn import pad_sequence
    smiles_padded = pad_sequence(
        smiles_tokens,
        batch_first=True,
        padding_value=PAD_IDX
    )
    
    # Create target (shifted right)
    tgt_input = smiles_padded[:, :-1]
    # Create a dummy mask for testing
    tgt_mask = torch.zeros(batch_size, tgt_input.size(1), tgt_input.size(1))
    
    # Test forward pass without embeddings
    print("  Testing Lightning forward pass without embeddings...")
    logits = lightning_module(h1_x, h1_mask, c13_x, c13_mask, tgt_input, tgt_mask)
    print(f"    Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, tgt_input.size(1), VOCAB_SIZE), "Logits shape mismatch"
    
    # Test forward pass with embeddings
    print("  Testing Lightning forward pass with embeddings...")
    logits, embeddings = lightning_module(h1_x, h1_mask, c13_x, c13_mask, tgt_input, tgt_mask, return_embeddings=True)
    print(f"    Logits shape: {logits.shape}")
    print(f"    Embeddings shape: {embeddings.shape}")
    assert logits.shape == (batch_size, tgt_input.size(1), VOCAB_SIZE), "Logits shape mismatch"
    assert embeddings.dim() == 3, "Embeddings should be 3D (batch, seq_len, d_model)"
    
    print("  ✓ Lightning module test passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Embedding Logging Implementation")
    print("=" * 60)
    
    try:
        test_embedding_extraction()
        test_lightning_module()
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
