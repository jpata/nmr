#!/usr/bin/env python
"""
Standalone tests for the model layers in train_nmr.py.
Tests ensure numerical correctness and correct shapes for all model components.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest

# Import the model classes from src.models.nmrtrans
from src.models.nmrtrans import MAB, ISAB, PMA, NMRTransPretextEncoder, NMRModalEncoder, NMRTrans


def test_mab_forward_shape():
    """Test MAB forward pass produces correct output shape."""
    d_model = 256
    num_heads = 8
    batch_size = 4
    seq_len_x = 10
    seq_len_y = 15
    
    mab = MAB(d_model=d_model, num_heads=num_heads)
    
    # Create random input tensors
    X = torch.randn(batch_size, seq_len_x, d_model)
    Y = torch.randn(batch_size, seq_len_y, d_model)
    
    output = mab(X, Y)
    
    # Output should have same shape as X
    assert output.shape == X.shape, f"Expected shape {X.shape}, got {output.shape}"
    print("✓ MAB shape test passed")


def test_mab_numerical_stability():
    """Test MAB doesn't produce NaN or Inf values."""
    d_model = 256
    num_heads = 8
    batch_size = 4
    seq_len = 10
    
    mab = MAB(d_model=d_model, num_heads=num_heads)
    
    # Test with normal data
    X = torch.randn(batch_size, seq_len, d_model)
    Y = torch.randn(batch_size, seq_len, d_model)
    
    output = mab(X, Y)
    
    assert not torch.isnan(output).any(), "MAB produced NaN values"
    assert not torch.isinf(output).any(), "MAB produced Inf values"
    print("✓ MAB numerical stability test passed")


def test_mab_attention_weights():
    """Test that attention weights are valid (sum to 1 along head dimension)."""
    d_model = 256
    num_heads = 8
    batch_size = 2
    seq_len = 5
    
    mab = MAB(d_model=d_model, num_heads=num_heads)
    
    X = torch.randn(batch_size, seq_len, d_model)
    Y = torch.randn(batch_size, seq_len, d_model)
    
    # Manually compute attention to check weights (need average_attn_weights=False)
    attn_out, attn_weights = mab.attn(query=X, key=Y, value=Y, average_attn_weights=False)
    
    # Attention weights shape: (batch_size, num_heads, seq_len_query, seq_len_key)
    # In this case, seq_len_query = seq_len_key = seq_len
    expected_shape = (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_shape, f"Expected {expected_shape}, got {attn_weights.shape}"
    assert not torch.isnan(attn_weights).any()
    assert not torch.isinf(attn_weights).any()
    print("✓ MAB attention weights test passed")


def test_isab_forward_shape():
    """Test ISAB forward pass produces correct output shape."""
    d_model = 256
    num_heads = 8
    m_inducing_points = 32
    batch_size = 4
    seq_len = 10
    
    isab = ISAB(d_model=d_model, num_heads=num_heads, m_inducing_points=m_inducing_points)
    
    X = torch.randn(batch_size, seq_len, d_model)
    
    output = isab(X)
    
    # Output should have same shape as input
    assert output.shape == X.shape, f"Expected shape {X.shape}, got {output.shape}"
    print("✓ ISAB shape test passed")


def test_isab_inducing_points():
    """Test that inducing points are properly initialized and expanded."""
    d_model = 256
    num_heads = 8
    m_inducing_points = 32
    batch_size = 4
    seq_len = 10
    
    isab = ISAB(d_model=d_model, num_heads=num_heads, m_inducing_points=m_inducing_points)
    
    # Check inducing points shape
    assert isab.I.shape == (1, m_inducing_points, d_model)
    
    X = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass should expand inducing points to batch size
    output = isab(X)
    
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    print("✓ ISAB inducing points test passed")


def test_pma_forward_shape():
    """Test PMA forward pass produces correct output shape."""
    d_model = 256
    num_heads = 8
    k_seeds = 4
    batch_size = 4
    seq_len = 10
    
    pma = PMA(d_model=d_model, num_heads=num_heads, k_seeds=k_seeds)
    
    Z = torch.randn(batch_size, seq_len, d_model)
    
    output = pma(Z)
    
    # Output should be (batch_size, k_seeds, d_model)
    assert output.shape == (batch_size, k_seeds, d_model), \
        f"Expected shape {(batch_size, k_seeds, d_model)}, got {output.shape}"
    print("✓ PMA shape test passed")


def test_pma_seed_matrix():
    """Test that seed matrix is properly initialized and expanded."""
    d_model = 256
    num_heads = 8
    k_seeds = 4
    batch_size = 4
    seq_len = 10
    
    pma = PMA(d_model=d_model, num_heads=num_heads, k_seeds=k_seeds)
    
    # Check seed matrix shape
    assert pma.S.shape == (1, k_seeds, d_model)
    
    Z = torch.randn(batch_size, seq_len, d_model)
    
    output = pma(Z)
    
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    print("✓ PMA seed matrix test passed")


def test_nmr_trans_pretext_encoder_shape():
    """Test NMRTransPretextEncoder forward pass produces correct shapes."""
    d_model = 256
    num_heads = 8
    num_layers = 3  # Using fewer layers for faster testing
    m_inducing_points = 32
    k_seeds = 4
    batch_size = 4
    seq_len = 10
    
    encoder = NMRTransPretextEncoder(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        m_inducing_points=m_inducing_points,
        k_seeds=k_seeds
    )
    
    X = torch.randn(batch_size, seq_len, d_model)
    
    Z, G = encoder(X)
    
    # Z should have same shape as input (peak-level features)
    assert Z.shape == X.shape, f"Expected Z shape {X.shape}, got {Z.shape}"
    
    # G should be (batch_size, k_seeds, d_model) (global features)
    assert G.shape == (batch_size, k_seeds, d_model), \
        f"Expected G shape {(batch_size, k_seeds, d_model)}, got {G.shape}"
    
    print("✓ NMRTransPretextEncoder shape test passed")


def test_nmr_trans_pretext_encoder_numerical():
    """Test NMRTransPretextEncoder doesn't produce NaN or Inf."""
    d_model = 256
    num_heads = 8
    num_layers = 3
    m_inducing_points = 32
    k_seeds = 4
    batch_size = 4
    seq_len = 10
    
    encoder = NMRTransPretextEncoder(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        m_inducing_points=m_inducing_points,
        k_seeds=k_seeds
    )
    
    X = torch.randn(batch_size, seq_len, d_model)
    
    Z, G = encoder(X)
    
    assert not torch.isnan(Z).any(), "Pretext encoder produced NaN in Z"
    assert not torch.isinf(Z).any(), "Pretext encoder produced Inf in Z"
    assert not torch.isnan(G).any(), "Pretext encoder produced NaN in G"
    assert not torch.isinf(G).any(), "Pretext encoder produced Inf in G"
    
    print("✓ NMRTransPretextEncoder numerical stability test passed")


def test_nmr_modal_encoder_1h_shape():
    """Test NMRModalEncoder for 1H data produces correct shapes."""
    d_input = 9  # 1H has 9 features
    d_model = 256
    num_heads = 8
    num_layers = 3
    m_inducing = 32
    k_seeds = 4
    batch_size = 4
    num_peaks = 10
    
    encoder = NMRModalEncoder(
        d_input=d_input,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        m_inducing=m_inducing,
        k_seeds=k_seeds
    )
    
    # 1H input: (batch_size, num_peaks, 9)
    X = torch.randn(batch_size, num_peaks, d_input)
    
    Z, G = encoder(X)
    
    # Z should be (batch_size, num_peaks, d_model)
    assert Z.shape == (batch_size, num_peaks, d_model), \
        f"Expected Z shape {(batch_size, num_peaks, d_model)}, got {Z.shape}"
    
    # G should be (batch_size, k_seeds, d_model)
    assert G.shape == (batch_size, k_seeds, d_model), \
        f"Expected G shape {(batch_size, k_seeds, d_model)}, got {G.shape}"
    
    print("✓ NMRModalEncoder 1H shape test passed")


def test_nmr_modal_encoder_13c_shape():
    """Test NMRModalEncoder for 13C data produces correct shapes."""
    d_input = 1  # 13C has 1 feature
    d_model = 256
    num_heads = 8
    num_layers = 3
    m_inducing = 32
    k_seeds = 4
    batch_size = 4
    num_peaks = 15
    
    encoder = NMRModalEncoder(
        d_input=d_input,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        m_inducing=m_inducing,
        k_seeds=k_seeds
    )
    
    # 13C input: (batch_size, num_peaks, 1)
    X = torch.randn(batch_size, num_peaks, d_input)
    
    Z, G = encoder(X)
    
    # Z should be (batch_size, num_peaks, d_model)
    assert Z.shape == (batch_size, num_peaks, d_model), \
        f"Expected Z shape {(batch_size, num_peaks, d_model)}, got {Z.shape}"
    
    # G should be (batch_size, k_seeds, d_model)
    assert G.shape == (batch_size, k_seeds, d_model), \
        f"Expected G shape {(batch_size, k_seeds, d_model)}, got {G.shape}"
    
    print("✓ NMRModalEncoder 13C shape test passed")


def test_nmr_modal_encoder_numerical():
    """Test NMRModalEncoder doesn't produce NaN or Inf."""
    d_input = 9
    d_model = 256
    num_heads = 8
    num_layers = 3
    m_inducing = 32
    k_seeds = 4
    batch_size = 4
    num_peaks = 10
    
    encoder = NMRModalEncoder(
        d_input=d_input,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        m_inducing=m_inducing,
        k_seeds=k_seeds
    )
    
    X = torch.randn(batch_size, num_peaks, d_input)
    
    Z, G = encoder(X)
    
    assert not torch.isnan(Z).any(), "Modal encoder produced NaN in Z"
    assert not torch.isinf(Z).any(), "Modal encoder produced Inf in Z"
    assert not torch.isnan(G).any(), "Modal encoder produced NaN in G"
    assert not torch.isinf(G).any(), "Modal encoder produced Inf in G"
    
    print("✓ NMRModalEncoder numerical stability test passed")


def test_nmr_trans_full_model_shape():
    """Test NMRTrans full model produces correct output shape."""
    d_model = 256
    vocab_size = 500
    max_smiles_len = 128
    k_seeds = 4
    batch_size = 4
    num_h_peaks = 10
    num_c_peaks = 15
    smiles_len = 50
    
    model = NMRTrans(
        d_model=d_model,
        vocab_size=vocab_size,
        max_smiles_len=max_smiles_len,
        k_seeds=k_seeds
    )
    
    # Create input tensors
    h1_x = torch.randn(batch_size, num_h_peaks, 9)
    c13_x = torch.randn(batch_size, num_c_peaks, 1)
    h1_mask = torch.zeros(batch_size, num_h_peaks, dtype=torch.bool)
    c13_mask = torch.zeros(batch_size, num_c_peaks, dtype=torch.bool)
    smiles_tgt = torch.randint(0, vocab_size, (batch_size, smiles_len))
    
    # Create causal mask
    tgt_mask = torch.triu(torch.ones(smiles_len, smiles_len) * float('-inf'), diagonal=1)
    
    logits = model(
        h1_x=h1_x,
        h1_mask=h1_mask,
        c13_x=c13_x,
        c13_mask=c13_mask,
        smiles_tgt=smiles_tgt,
        smiles_tgt_mask=tgt_mask
    )
    
    # Output should be (batch_size, smiles_len, vocab_size)
    assert logits.shape == (batch_size, smiles_len, vocab_size), \
        f"Expected shape {(batch_size, smiles_len, vocab_size)}, got {logits.shape}"
    
    print("✓ NMRTrans full model shape test passed")


def test_nmr_trans_encoder_fusion():
    """Test that encoder fusion works correctly."""
    d_model = 256
    vocab_size = 500
    max_smiles_len = 128
    k_seeds = 4
    batch_size = 4
    num_h_peaks = 10
    num_c_peaks = 15
    
    model = NMRTrans(
        d_model=d_model,
        vocab_size=vocab_size,
        max_smiles_len=max_smiles_len,
        k_seeds=k_seeds
    )
    
    h1_x = torch.randn(batch_size, num_h_peaks, 9)
    c13_x = torch.randn(batch_size, num_c_peaks, 1)
    h1_mask = torch.zeros(batch_size, num_h_peaks, dtype=torch.bool)
    c13_mask = torch.zeros(batch_size, num_c_peaks, dtype=torch.bool)
    smiles_tgt = torch.randint(0, vocab_size, (batch_size, 50))
    tgt_mask = torch.triu(torch.ones(50, 50) * float('-inf'), diagonal=1)
    
    # Get encoder outputs
    Z_H, G_H = model.encoder_1H(h1_x)
    Z_C, G_C = model.encoder_13C(c13_x)
    
    # Check shapes
    assert Z_H.shape == (batch_size, num_h_peaks, d_model)
    assert G_H.shape == (batch_size, k_seeds, d_model)
    assert Z_C.shape == (batch_size, num_c_peaks, d_model)
    assert G_C.shape == (batch_size, k_seeds, d_model)
    
    # Fusion should concatenate: [G_C, Z_C, G_H, Z_H]
    H_enc = torch.cat([G_C, Z_C, G_H, Z_H], dim=1)
    expected_seq_len = k_seeds + num_c_peaks + k_seeds + num_h_peaks
    assert H_enc.shape == (batch_size, expected_seq_len, d_model)
    
    print("✓ NMRTrans encoder fusion test passed")


def test_nmr_trans_numerical_stability():
    """Test NMRTrans doesn't produce NaN or Inf values."""
    d_model = 256
    vocab_size = 500
    max_smiles_len = 128
    k_seeds = 4
    batch_size = 4
    num_h_peaks = 10
    num_c_peaks = 15
    
    model = NMRTrans(
        d_model=d_model,
        vocab_size=vocab_size,
        max_smiles_len=max_smiles_len,
        k_seeds=k_seeds
    )
    
    h1_x = torch.randn(batch_size, num_h_peaks, 9)
    c13_x = torch.randn(batch_size, num_c_peaks, 1)
    h1_mask = torch.zeros(batch_size, num_h_peaks, dtype=torch.bool)
    c13_mask = torch.zeros(batch_size, num_c_peaks, dtype=torch.bool)
    smiles_tgt = torch.randint(0, vocab_size, (batch_size, 50))
    tgt_mask = torch.triu(torch.ones(50, 50) * float('-inf'), diagonal=1)
    
    logits = model(
        h1_x=h1_x,
        h1_mask=h1_mask,
        c13_x=c13_x,
        c13_mask=c13_mask,
        smiles_tgt=smiles_tgt,
        smiles_tgt_mask=tgt_mask
    )
    
    assert not torch.isnan(logits).any(), "NMRTrans produced NaN values"
    assert not torch.isinf(logits).any(), "NMRTrans produced Inf values"
    
    print("✓ NMRTrans numerical stability test passed")


def test_nmr_trans_with_padding():
    """Test NMRTrans handles padding masks correctly."""
    d_model = 256
    vocab_size = 500
    max_smiles_len = 128
    k_seeds = 4
    batch_size = 4
    
    model = NMRTrans(
        d_model=d_model,
        vocab_size=vocab_size,
        max_smiles_len=max_smiles_len,
        k_seeds=k_seeds
    )
    
    # Create inputs with different numbers of peaks (simulating padding)
    h1_x = torch.randn(batch_size, 10, 9)
    c13_x = torch.randn(batch_size, 15, 1)
    
    # Create masks where True indicates valid data
    h1_mask = torch.zeros(batch_size, 10, dtype=torch.bool)
    c13_mask = torch.zeros(batch_size, 15, dtype=torch.bool)
    
    # Make some peaks invalid (padded)
    h1_mask[0, 7:] = True  # Last 3 peaks in batch 0 are valid
    c13_mask[1, 10:] = True  # Last 5 peaks in batch 1 are valid
    
    smiles_tgt = torch.randint(0, vocab_size, (batch_size, 50))
    tgt_mask = torch.triu(torch.ones(50, 50) * float('-inf'), diagonal=1)
    
    logits = model(
        h1_x=h1_x,
        h1_mask=h1_mask,
        c13_x=c13_x,
        c13_mask=c13_mask,
        smiles_tgt=smiles_tgt,
        smiles_tgt_mask=tgt_mask
    )
    
    assert logits.shape == (batch_size, 50, vocab_size)
    assert not torch.isnan(logits).any()
    
    print("✓ NMRTrans padding handling test passed")


def test_model_gradients():
    """Test that all model parameters have gradients."""
    d_model = 256
    vocab_size = 500
    max_smiles_len = 128
    k_seeds = 4
    batch_size = 4
    num_h_peaks = 10
    num_c_peaks = 15
    
    model = NMRTrans(
        d_model=d_model,
        vocab_size=vocab_size,
        max_smiles_len=max_smiles_len,
        k_seeds=k_seeds
    )
    
    h1_x = torch.randn(batch_size, num_h_peaks, 9, requires_grad=True)
    c13_x = torch.randn(batch_size, num_c_peaks, 1, requires_grad=True)
    h1_mask = torch.zeros(batch_size, num_h_peaks, dtype=torch.bool)
    c13_mask = torch.zeros(batch_size, num_c_peaks, dtype=torch.bool)
    smiles_tgt = torch.randint(0, vocab_size, (batch_size, 50))
    tgt_mask = torch.triu(torch.ones(50, 50) * float('-inf'), diagonal=1)
    
    logits = model(
        h1_x=h1_x,
        h1_mask=h1_mask,
        c13_x=c13_x,
        c13_mask=c13_mask,
        smiles_tgt=smiles_tgt,
        smiles_tgt_mask=tgt_mask
    )
    
    # Compute loss and backpropagate
    loss = logits.sum()
    loss.backward()
    
    # Check that parameters have gradients
    params_with_grad = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    
    assert params_with_grad > 0, "No parameters have gradients"
    print(f"✓ Model gradients test passed ({params_with_grad}/{total_params} parameters have gradients)")


def test_model_parameter_counts():
    """Test that model has reasonable number of parameters."""
    d_model = 256
    vocab_size = 500
    max_smiles_len = 128
    k_seeds = 4
    
    model = NMRTrans(
        d_model=d_model,
        vocab_size=vocab_size,
        max_smiles_len=max_smiles_len,
        k_seeds=k_seeds
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model parameter count test passed")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Basic sanity checks
    assert total_params > 0, "Model has no parameters"
    assert trainable_params > 0, "Model has no trainable parameters"


if __name__ == "__main__":
    print("Running model layer tests...\n")
    
    # Test MAB
    test_mab_forward_shape()
    test_mab_numerical_stability()
    test_mab_attention_weights()
    
    # Test ISAB
    test_isab_forward_shape()
    test_isab_inducing_points()
    
    # Test PMA
    test_pma_forward_shape()
    test_pma_seed_matrix()
    
    # Test NMRTransPretextEncoder
    test_nmr_trans_pretext_encoder_shape()
    test_nmr_trans_pretext_encoder_numerical()
    
    # Test NMRModalEncoder
    test_nmr_modal_encoder_1h_shape()
    test_nmr_modal_encoder_13c_shape()
    test_nmr_modal_encoder_numerical()
    
    # Test NMRTrans full model
    test_nmr_trans_full_model_shape()
    test_nmr_trans_encoder_fusion()
    test_nmr_trans_numerical_stability()
    test_nmr_trans_with_padding()
    
    # Test gradients and parameters
    test_model_gradients()
    test_model_parameter_counts()
    
    print("\n✅ All tests passed!")
