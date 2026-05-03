# Human Uncertainty for Image Classification

A deep learning project exploring human uncertainty in image classification tasks, leveraging ensemble learning techniques and uncertainty quantification methods.

## Overview

This project implements and evaluates multiple deep neural network architectures for image classification on CIFAR-10 dataset, with a focus on capturing and modeling human uncertainty. The system incorporates soft label probabilities from human annotations to improve model robustness and uncertainty estimates.

## Features

- **Multiple Model Architectures**:
  - ResNet (20-layer and basic 110-layer variants)
  - VGG-16 with Batch Normalization
  - LightCNN

- **Uncertainty Quantification Methods**:
  - KL Divergence loss
  - Jensen-Shannon Divergence (JSD)
  - Entropy Penalty

- **Soft Label Learning**: 
  - Utilizes CIFAR-10H dataset with human probability distributions
  - Improves model calibration and uncertainty awareness

- **Comprehensive Evaluation**:
  - Accuracy metrics
  - Loss analysis
  - Visualization of results
  - Model checkpoints with best accuracy tracking

## Project Structure

```
├── main.py                          # Entry point for training and evaluation
├── README.md                        # This file
├── data/                            # Dataset storage
│   ├── cifar10h-probs.npy          # CIFAR-10H human probability distributions
│   └── cifar-10-batches-py/        # CIFAR-10 dataset batches
├── member1_dataset/                 # Dataset handling module
│   └── dataset.py                  # CIFAR-10 dataset loading and preprocessing
├── member2_model/                   # Model architecture definitions
│   └── models.py                   # ResNet, VGG16, LightCNN implementations
├── member3_training/                # Training module
│   └── train.py                    # Training loop and optimization
├── member4_evaluation/              # Evaluation module
│   └── evaluate.py                 # Model evaluation and metrics
├── results/                         # Training results and checkpoints
│   ├── *.json                      # Result summaries and loss tracking
│   ├── checkpoints/                # Trained model checkpoints
│   └── plots/                      # Visualization outputs
└── evaluation_results/              # Evaluation outputs
    ├── evaluation_summary.csv       # Summary statistics
    ├── evaluation_summary.json      # JSON formatted results
    └── plots/                       # Result visualizations
```

## Dataset

- **CIFAR-10**: 60,000 32x32 color images across 10 classes
- **CIFAR-10H**: Human uncertainty annotations with probability distributions over classes
  - Incorporates multiple human labelers' opinions
  - Provides soft targets for training

## Models Trained

The following model configurations have been evaluated:

1. **ResNet20** - Compact 20-layer ResNet
   - Variants: KL, JSD, Entropy Penalty losses
   
2. **ResNet Basic 110** - 110-layer basic ResNet
   - Variants: KL, Entropy Penalty losses
   
3. **VGG16-BN** - VGG16 with Batch Normalization
   - Variant: KL loss
   
4. **LightCNN** - Lightweight CNN for mobile/embedded deployment
   - Variants: KL, JSD, Entropy Penalty losses

## Loss Functions

- **KL Divergence**: Kullback-Leibler divergence between predictions and soft targets
- **JSD (Jensen-Shannon)**: Symmetric divergence measure
- **Entropy Penalty**: Reduces prediction uncertainty with entropy regularization

## Getting Started

### Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sreekruthy/human_uncertainity_for_image_classification.git
cd human_uncertainity_for_image_classification
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib
```

### Usage

Run the main script to train and evaluate models:

```bash
python main.py
```

Or run individual modules:

```bash
# Train models
python member3_training/train.py

# Evaluate models
python member4_evaluation/evaluate.py
```

## Results

Training results are stored in the `results/` directory including:
- Loss summaries for each model architecture
- Best accuracy checkpoints
- Performance metrics across different loss functions

Evaluation results with visualizations are available in `evaluation_results/`.

## Key Insights

1. Soft label learning improves model calibration compared to hard labels
2. Entropy penalty effectively reduces prediction uncertainty
3. Different model architectures show varying sensitivity to different loss functions
4. Ensemble uncertainty estimates correlate with classification difficulty

## Contributing

This project was developed as part of a collaborative team effort:
- Member 1: Dataset handling and preprocessing
- Member 2: Model architecture implementation
- Member 3: Training framework and optimization
- Member 4: Evaluation and analysis

## License

This project is provided as-is for educational and research purposes.

## References

- Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical report.
- Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., & Griffiths, T. L. (2021). Using large-scale experiments and machine learning to discover theories of human decision-making. Science, 372(6547), 1209-1214.

---

**Last Updated**: May 2026
