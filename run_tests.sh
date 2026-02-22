#!/bin/bash
# Script to run tests inside singularity container

IMG="/home/software/singularity/pytorch.simg:2026-02-04"

cd /home/joosep/nmr

echo "Running tests inside singularity container..."
singularity exec --nv $IMG python3 test_data_loading.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Tests passed successfully!"
else
    echo "✗ Tests failed!"
fi

exit $EXIT_CODE
