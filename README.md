# Adaptive Multi-Branch QCNN

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-000000)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research repository exploring hybrid classical–quantum architectures for image classification. This project investigates how quantum convolutional layers, multi-branch feature extraction, attention-based fusion, and adaptive routing can be combined to build expressive yet trainable models on standard vision benchmarks.

The work progresses from classical baselines through intermediate quantum experiments toward a final **Adaptive Multi-Branch QCNN** design. Each stage documents architectural choices, training dynamics, and comparative evaluation to provide a transparent record of the research process.

## Key Highlights

- End-to-end experimentation from classical baselines to hybrid quantum models
- Multi-branch quantum convolutional architectures with feature fusion
- Attention mechanisms for adaptive inter-branch integration
- Systematic evaluation across model variants and ablation settings
- Modular source layout alongside a complete research notebook

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebook/          # Complete research notebook (full experimentation journey)
├── src/               # Modular implementation of models, data, training, and evaluation
├── figures/           # Generated plots and visualizations
├── results/           # Metrics, checkpoints, and experiment outputs
└── paper/             # Manuscript, supplementary materials, and related assets
```

> **Note:** The notebook in `notebook/` contains the complete research journey — including baselines, intermediate experiments, quantum architectures, fusion strategies, and final model development. The `src/` directory provides a clean, modular implementation suitable for reuse and extension.
## Dataset

This project uses the Brain Tumor MRI Dataset.

Download:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

After downloading, extract it as:

data/
└── raw/
    ├── Training/
    └── Testing/
## Dataset

This project uses the Brain Tumor MRI Dataset.

Download:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

After downloading, extract it as:

data/
└── raw/
    ├── Training/
    └── Testing/
    
    
    Hybrid QCNN (Angle + Amplitude Encoding, 5 Epochs)
This experiment combines a classical CNN feature extractor with a variational quantum circuit. Images are compressed into an 8-dimensional latent representation before being encoded into an 8-qubit quantum circuit using angle embedding. The circuit consists of six variational layers with RY and RZ rotations connected by circular CNOT entanglement. The model was trained for 5 epochs to evaluate the effectiveness of hybrid quantum-classical learning for multi-class brain tumor MRI classification.