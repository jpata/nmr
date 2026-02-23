# Proposed Directory Structure for NMR ML Project

```
nmr/
├── data/
│   ├── raw/                  # Original, immutable data dumps
│   ├── processed/            # Processed data ready for modeling
│   ├── external/             # Third-party datasets
│   └── temp/                 # Temporary working files
│
├── models/
│   ├── base/                # Base model architectures
│   ├── trained/              # Checkpoints from training runs
│   └── exports/              # Export models for inference
│
├── src/
│   ├── data/                # Data loading and processing
│   │   ├── loaders.py        # Data loaders and datasets
│   │   ├── preprocessing.py  # Data preprocessing functions
│   │   └── validation.py     # Data validation scripts
│   │
│   ├── models/              # Model architectures
│   │   ├── components.py     # Model components (MAB, ISAB, etc.)
│   │   ├── nmrtrans.py       # Main NMRTrans model
│   │   └── lightning.py      # PyTorch Lightning modules
│   │
│   ├── utils/               # Utility functions
│   │   ├── tokenizer.py      # SMILES tokenizer
│   │   ├── logging.py        # Custom logging utilities
│   │   └── helpers.py        # Helper functions
│   │
│   └── train.py             # Main training script
│
├── tests/
│   ├── unit/                # Unit tests for individual components
│   ├── integration/          # Integration tests
│   └── data/                # Data validation tests
│
├── configs/
│   ├── data_config.yaml     # Data configuration
│   ├── model_config.yaml    # Model hyperparameters
│   └── train_config.yaml    # Training configuration
│
├── notebooks/
│   ├── exploration/         # Data exploration notebooks
│   ├── analysis/            # Results analysis
│   └── experiments/         # Experiment tracking
│
├── scripts/
│   ├── run_training.sh      # Training execution script
│   ├── run_tests.sh         # Test execution script
│   └── data_processing.sh   # Data processing pipeline
│
├── logs/
│   ├── training/            # Training logs
│   └── tests/               # Test logs
│
├── docs/
│   ├── README.md            # Project overview
│   ├── setup.md             # Setup instructions
│   └── architecture.md      # Architecture documentation
│
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
├── wrapper.sh               # Singularity execution wrapper
└── run_tests.sh             # Test runner script
```

## Key Improvements:

1. **Separation of Concerns**: Models, data, and utilities are in separate directories
2. **Modular Architecture**: Components can be tested and developed independently
3. **Reproducibility**: Configuration files separate from code
4. **Scalability**: Easy to add new models, data sources, or experiments
5. **Maintainability**: Clear structure for documentation and onboarding

## Migration Plan:

1. Split `train_nmr.py` into logical components in `src/`
2. Move test files to `tests/` directory with proper organization
3. Create configuration files for hyperparameters
4. Add documentation for each component
5. Set up proper data versioning in `data/`
