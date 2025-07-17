# EdgeSpark - 2D Fragment Matching

A deep learning project for matching irregular 2D fragments based on edge information.

## 📁 Project Structure

```
EdgeSpark/
├── PROJECT_README.md                 # This file
├── CLAUDE.md                         # Project instructions for AI assistant
├── EXPERIMENT_SUMMARY.md             # Detailed analysis and results
├── main.py                           # Main entry point
├── pyproject.toml                    # Package configuration
├── dataset/                          # Dataset pickle files
│   ├── train_set.pkl
│   ├── valid_set.pkl
│   └── test_set.pkl
├── original_code/                    # Original implementation (reference)
│   ├── network_improved.py          # Original complex network
│   ├── train.py                      # Original training script
│   └── dataset_loader.py            # Original data loader
├── simplified_approach/              # Simplified implementation
│   ├── network_simple.py            # Simplified network architecture
│   ├── dataset_simple.py            # Optimized data processing
│   └── train_simple.py              # Simplified training script
├── final_approach/                   # Best implementation
│   ├── final_approach.py            # **RECOMMENDED** - Best network & training
│   ├── network_minimal.py           # Minimal network (for comparison)
│   └── train_minimal.py             # Minimal training script
├── analysis_tools/                  # Analysis and debugging tools
│   ├── debug_data.py                # Data quality analysis
│   ├── quick_test.py                # Quick functionality test
│   └── run_simple_test.py           # Comprehensive test suite
└── archive/                         # Archived experiments
    └── experiments/                  # Previous training results
```

## 🚀 Quick Start

### Prerequisites
- Python ≥3.13
- CUDA 11.8 support
- `uv` package manager

### Installation
```bash
# Install dependencies
uv sync

# Quick test
uv run python analysis_tools/quick_test.py
```

### Training (Recommended)
```bash
# Use the best implementation
uv run python final_approach/final_approach.py

# Or with custom parameters
uv run python final_approach/final_approach.py --epochs 30 --lr 0.001 --batch-size 32
```

### Alternative Approaches
```bash
# Simplified approach
uv run python simplified_approach/train_simple.py --epochs 20

# Minimal approach
uv run python final_approach/train_minimal.py --epochs 15
```

## 📊 Results Summary

| Approach | Accuracy | F1-Score | AUC | Parameters | Status |
|----------|----------|----------|-----|------------|--------|
| Original Complex | 50.0% | 0.0000 | 0.5000 | ~500K | ❌ Failed |
| Simplified | 59.85% | 0.5926 | 0.5909 | 224K | ⚠️ Partial |
| Minimal | 50.0% | 0.0000 | 0.5000 | 87K | ❌ Failed |
| **Final (Recommended)** | **60.95%** | **0.6492** | **0.6100** | **1.1M** | ✅ **Best** |

## 🔍 Key Findings

1. **Data Quality is Critical**: Simple geometric features cannot distinguish matching pairs
2. **Network Architecture**: Edge-specific encoders outperform general architectures  
3. **Training Strategy**: Learning rate 0.001 with StepLR scheduler works best
4. **Feature Engineering**: Concatenation + difference + dot product similarity is effective

## 📖 Detailed Analysis

For complete analysis, problem diagnosis, and solution details, see:
- **[EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)** - Comprehensive analysis and results

## 🛠️ Development

### Data Analysis
```bash
# Analyze data quality
uv run python analysis_tools/debug_data.py

# Run comprehensive tests
uv run python analysis_tools/run_simple_test.py
```

### Custom Training
```bash
# Modify hyperparameters in the respective training files
# Or create custom configurations in the training scripts
```

## 💡 Future Improvements

1. **Graph Neural Networks**: Model fragments as graph structures
2. **Attention Mechanisms**: More sophisticated attention for shape matching
3. **Multi-scale Features**: Capture features at different scales
4. **Data Augmentation**: Smarter augmentation preserving edge information
5. **Ensemble Methods**: Combine multiple models for better performance

## 🎯 Best Practices

Based on experimental results:

1. **Use `final_approach/final_approach.py`** for best results
2. **Learning rate**: 0.001 with StepLR scheduler
3. **Batch size**: 32-64 for optimal stability
4. **Data preprocessing**: Hard negative mining + light augmentation
5. **Feature fusion**: Multiple fusion strategies work better than single

## 📋 Requirements

- PyTorch 2.7.1+
- CUDA 11.8+
- scikit-learn (for evaluation metrics)
- matplotlib (for visualization)
- numpy

## 🤝 Contributing

This project demonstrates a complete deep learning debugging workflow:
1. Problem identification and diagnosis
2. Systematic experimentation
3. Data quality analysis
4. Architecture optimization
5. Training strategy refinement

## 📄 License

See project documentation for licensing information.

---

*For AI assistants: See CLAUDE.md for detailed project context and instructions.*