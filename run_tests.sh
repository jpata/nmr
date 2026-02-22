#!/bin/bash
# Script to run all tests inside singularity container

IMG="/home/software/singularity/pytorch.simg:2026-02-04"

cd /home/joosep/nmr

echo "=========================================="
echo "Running Data Loading Tests..."
echo "=========================================="
singularity exec --nv $IMG python3 test_data_loading.py
DATA_LOADING_EXIT=$?

echo ""
echo "=========================================="
echo "Running Model Layer Tests..."
echo "=========================================="
singularity exec --nv $IMG python3 test_model_layers.py
MODEL_LAYERS_EXIT=$?

echo ""
echo "=========================================="
echo "Running Tokenizer Tests..."
echo "=========================================="
singularity exec --nv $IMG python3 test_tokenizer.py
TOKENIZER_EXIT=$?

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="

if [ $DATA_LOADING_EXIT -eq 0 ]; then
    echo "✓ Data Loading Tests: PASSED"
else
    echo "✗ Data Loading Tests: FAILED"
fi

if [ $MODEL_LAYERS_EXIT -eq 0 ]; then
    echo "✓ Model Layer Tests: PASSED"
else
    echo "✗ Model Layer Tests: FAILED"
fi

if [ $TOKENIZER_EXIT -eq 0 ]; then
    echo "✓ Tokenizer Tests: PASSED"
else
    echo "✗ Tokenizer Tests: FAILED"
fi

# Overall exit code
if [ $DATA_LOADING_EXIT -eq 0 ] && [ $MODEL_LAYERS_EXIT -eq 0 ] && [ $TOKENIZER_EXIT -eq 0 ]; then
    echo ""
    echo "✓✓✓ ALL TESTS PASSED! ✓✓✓"
    exit 0
else
    echo ""
    echo "✗✗✗ SOME TESTS FAILED! ✗✗✗"
    exit 1
fi
