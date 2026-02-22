# Testing Guide for NMR Data Loading and Model Layers

This document describes how to run the standalone tests for both data loading functionality and model layers in `train_nmr.py`.

## Running Tests

### Using the Run Script

The easiest way to run tests is using the provided run script:

```bash
./run_tests.sh
```

This script will:
1. Run the tests inside the singularity container
2. Display the test results
3. Return an appropriate exit code (0 for success, 1 for failure)

### Manual Execution

You can also run the tests manually inside the singularity container:

**Run Data Loading Tests:**
```bash
cd /home/joosep/nmr
singularity exec /home/software/singularity/pytorch.simg:2026-02-04 python3 test_data_loading.py
```

**Run Model Layer Tests:**
```bash
cd /home/joosep/nmr
singularity exec /home/software/singularity/pytorch.simg:2026-02-04 python3 test_model_layers.py
```

**Run Both Test Suites:**
```bash
cd /home/joosep/nmr
./run_tests.sh
```

## Test Coverage

The test suite covers the following components across two test files:

### test_data_loading.py - Data Loading and Preprocessing Tests

#### 1. Peak Parsing Functions
- `parse_1H_peaks()` - Tests for 1H NMR peak parsing
  - Single and multiple peaks
  - J-coupling sorting and padding
  - String input handling
  - Empty data handling

- `parse_13C_peaks()` - Tests for 13C NMR peak parsing
  - Single and multiple peaks
  - Range midpoint calculation
  - String input handling
  - Empty data handling

### 2. Dataset Components
- `SMILESTokenizer` - Tests for SMILES tokenization
  - Basic tokenization
  - Special character handling
  - Encoding/decoding round-trip
  - Unknown token handling

- `NMRDataset` - Tests for the PyTorch Dataset class
  - Dataset length calculation
  - Item retrieval
  - Multi-modality handling
  - Empty dataset handling

### 3. Data Loading Pipeline
- `nmr_collate_fn()` - Tests for the custom collate function
  - Batch collation
  - Mask generation
  - Padding behavior

- `get_dataloader()` - Tests for DataLoader creation
  - DataLoader properties
  - Batch iteration
  - Data integrity

### 4. Edge Cases
- Empty DataFrames
- Single modality per SMILES
- None value handling
- Empty list handling

### 5. Integration Tests
- Full pipeline testing with random data generation
- End-to-end validation of all components working together

### test_model_layers.py - Model Layer Tests

#### 1. Multihead Attention Block (MAB)
- Forward pass shape validation
- Numerical stability (no NaN/Inf values)
- Attention weight validation

#### 2. Induced Set Attention Block (ISAB)
- Forward pass shape validation
- Inducing points initialization and expansion
- Numerical stability

#### 3. Pooling by Multihead Attention (PMA)
- Forward pass shape validation
- Seed matrix initialization and expansion
- Numerical stability

#### 4. NMRTransPretextEncoder
- Shape validation for both equivariant (Z) and invariant (G) outputs
- Numerical stability across multiple ISAB layers
- Correct handling of variable-length inputs

#### 5. NMRModalEncoder
- Shape validation for 1H (9 features) and 13C (1 feature) inputs
- Input projection and feature transformation
- Numerical stability

#### 6. NMRTrans (Full Model)
- End-to-end shape validation
- Encoder fusion (concatenation of G_C, Z_C, G_H, Z_H)
- Numerical stability with various input sizes
- Padding mask handling
- Gradient flow validation (all parameters receive gradients)
- Parameter count validation

Running Data Loading Tests for train_nmr.py
============================================================

Testing parse_1H_peaks...
  ✓ Single peak parsing works
  ✓ Multiple peaks parsing works
  ✓ J-coupling sorting works
  ✓ Empty data handling works
  ✓ String input parsing works
  ✓ Zero-padding for missing J-couplings works
✓ All parse_1H_peaks tests passed!

... (more test output) ...

============================================================
✓ ALL TESTS PASSED!
============================================================
```
=======
## Test Output

When tests run successfully using the run script, you'll see output like:

```
==========================================
Running Data Loading Tests...
==========================================
============================================================
Running Data Loading Tests for train_nmr.py
============================================================

Testing parse_1H_peaks...
  ✓ Single peak parsing works
  ✓ Multiple peaks parsing works
  ✓ J-coupling sorting works
  ✓ Empty data handling works
  ✓ String input parsing works
  ✓ Zero-padding for missing J-couplings works
✓ All parse_1H_peaks tests passed!

... (more data loading test output) ...

============================================================
✓ ALL TESTS PASSED!
============================================================

==========================================
Running Model Layer Tests...
==========================================
Running model layer tests...

✓ MAB shape test passed
✓ MAB numerical stability test passed
✓ MAB attention weights test passed
✓ ISAB shape test passed
✓ ISAB inducing points test passed
✓ PMA shape test passed
✓ PMA seed matrix test passed
✓ NMRTransPretextEncoder shape test passed
✓ NMRTransPretextEncoder numerical stability test passed
✓ NMRModalEncoder 1H shape test passed
✓ NMRModalEncoder 13C shape test passed
✓ NMRModalEncoder numerical stability test passed
✓ NMRTrans full model shape test passed
✓ NMRTrans encoder fusion test passed
✓ NMRTrans numerical stability test passed
✓ NMRTrans padding handling test passed
✓ Model gradients test passed (442/442 parameters have gradients)
✓ Model parameter count test passed
  Total parameters: 27,247,092
  Trainable parameters: 27,247,092

✅ All tests passed!

==========================================
Test Summary
==========================================
✓ Data Loading Tests: PASSED
✓ Model Layer Tests: PASSED

✓✓✓ ALL TESTS PASSED! ✓✓✓
```
============================================================
Running Data Loading Tests for train_nmr.py
============================================================

Testing parse_1H_peaks...
  ✓ Single peak parsing works
  ✓ Multiple peaks parsing works
  ✓ J-coupling sorting works
  ✓ Empty data handling works
  ✓ String input parsing works
  ✓ Zero-padding for missing J-couplings works
✓ All parse_1H_peaks tests passed!

... (more test output) ...

============================================================
✓ ALL TESTS PASSED!
============================================================
```

## Troubleshooting

### Tests Fail with Import Errors

If you see import errors, make sure:
1. You're running from the `/home/joosep/nmr` directory
2. The singularity container has the required dependencies (pandas, torch, sklearn)

### Tests Fail with Data Format Errors

The tests expect specific data formats:
- 1H NMR data: tuples of `(multiplicity, j_couplings_str, integration, shift1, shift2)`
- 13C NMR data: tuples of `(shift, multiplicity, integration)`
- J-couplings should be formatted as string lists like `["7.2", "7.3"]`

### Tests Fail with Assertion Errors

Check the specific assertion that failed. The error message should indicate what went wrong. Common issues:
- Data shape mismatches
- Incorrect tensor values
- Missing expected data

## Adding New Tests

To add new tests:

1. Add a new test function following the pattern of existing tests
2. Include appropriate assertions to validate the behavior
3. Add descriptive print statements to show what's being tested
4. Call the new test function in the `main()` function

Example:

```python
def test_new_feature():
    """Test description"""
    print("Testing new feature...")
    
    # Test logic here
    assert expected_result == actual_result, "Error message"
    
    print("  ✓ Test passed")

# In main():
# test_new_feature()
```

## Continuous Integration

To integrate these tests into a CI/CD pipeline, you can run:

### Run All Tests
```bash
./run_tests.sh
```

This will run both test suites and provide a summary.

### Run Individual Test Suites

**Data Loading Tests:**
```bash
singularity exec /home/software/singularity/pytorch.simg:2026-02-04 python3 test_data_loading.py
```

**Model Layer Tests:**
```bash
singularity exec /home/software/singularity/pytorch.simg:2026-02-04 python3 test_model_layers.py
```

The exit code will indicate success (0) or failure (1).
