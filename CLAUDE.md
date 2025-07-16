# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EdgeSpark is a 2D fragment matching project that aims to match irregular 2D fragments by finding potentially adjacent fragments in a fragment collection. The project uses a brute-force sampling approach with deep learning for fragment matching.

## Dataset Structure

The project uses three pickle datasets located in `./dataset/`:
- `train_set.pkl` - Training dataset
- `test_set.pkl` - Testing dataset  
- `valid_set.pkl` - Validation dataset

Each dataset contains:
- `full_pcd_all`: Edge point cloud data for each fragment
- `GT_pairs`: Ground truth matching fragment pairs
- `source_ind`: Index of matching points in source fragments
- `target_ind`: Index of matching points in target fragments

## Development Environment

### Package Management
- **Primary**: Uses `uv` package manager with `pyproject.toml`
- **Legacy**: Contains `requirments.yaml` for conda environment (from previous project)
- **Python**: Requires Python ≥3.13
- **CUDA**: Configured for CUDA 11.8 support

### Key Dependencies
- PyTorch with CUDA support
- NumPy, Matplotlib for data processing and visualization
- Various deep learning libraries (see pyproject.toml)

## Project Structure

```
EdgeSpark/
├── main.py                    # Simple entry point
├── dataset/                   # Dataset pickle files
├── data-analyze/              # Data analysis scripts
├── Data Generation Code/      # External project code for data generation
├── PairingNet Code/          # External project code (reference implementation)
└── pyproject.toml            # Package configuration
```

## Core Algorithm

The project implements a brute-force sampling approach:

1. **Random Sampling**: Sample n1, n2 fragment segments from two fragments based on edge length
2. **Feature Encoding**: Encode sampled segments 
3. **Similarity Computation**: Calculate segment similarity using n1 × n2 × d matrix
4. **Matching**: Use deep learning (comprehensive module) to determine fragment pair matching probability

### Key Advantage
Random sampling with sufficient iterations can sample segments with minimal offset, reducing the impact of alignment issues in traditional fixed-interval segmentation.

## Network Architecture

- **Transformer-based**: Uses Transformer architecture for the n1 × n2 sequence
- **Self-attention**: Natural fit for pairwise matching process
- **Loss Function**: Binary cross-entropy for end-to-end fragment matching

## Important Notes

- `Data Generation Code/` and `PairingNet Code/` contain external reference code from another project
- These directories are related to the dataset but not core to this project's implementation
- The main project code is expected to be developed in the root directory and other folders

## Development Commands

```bash
# Install dependencies
uv sync

# Run main entry point
python main.py

# For legacy conda environment (if needed)
conda env create -f requirments.yaml
```