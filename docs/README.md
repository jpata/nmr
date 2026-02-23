# NMRTrans Project

A PyTorch implementation of NMRTrans for predicting molecular structures from NMR spectra.

## Project Structure

```
nmr/
├── data/                  # Data files
├── models/                # Model architectures and checkpoints
├── src/                   # Source code
│   ├── data/              # Data loading and processing
│   ├── models/            # Model architectures
│   ├── utils/             # Utility functions
│   └── train.py           # Main training script
├── tests/                 # Test suites
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── data/              # Data validation tests
├── configs/               # Configuration files
├── notebooks/             # Jupyter notebooks
├── scripts/               # Execution scripts
├── logs/                  # Log files
├── docs/                  # Documentation
└── .gitignore             # Git ignore rules
```

## Quick Start

### Running Tests

```bash
./scripts/run_tests.sh
```

### Training the Model

```bash
# Local execution
./scripts/wrapper.sh python src/train.py

# GPU execution via SLURM
sbatch scripts/wrapper.sh python src/train.py
```

## Key Components

- **NMRTrans Model**: Set transformer architecture for NMR spectrum analysis
- **SMILESTokenizer**: Chemical-aware tokenizer for SMILES strings
- **Data Pipeline**: Efficient loading and preprocessing of NMR data
- **PyTorch Lightning**: Modular training framework with logging and callbacks

## Configuration

Training parameters are configured directly in the training script. For advanced use cases, consider creating YAML configuration files in the `configs/` directory.

## Data Format

The project expects NMR data in Parquet format with the following columns:
- `SMILES`: Molecular structure in SMILES notation
- `NMR_type`: Type of NMR spectrum ('1H NMR' or '13C NMR')
- `NMR_processed`: Processed peak data

## Execution Environment

This project uses Singularity containers for reproducible execution. The wrapper script handles container execution automatically.

See `docs/instructions.md` for detailed execution instructions.
