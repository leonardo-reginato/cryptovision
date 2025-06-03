# 🐟 CryptoVision

Welcome to **CryptoVision**! This project focuses on building a hierarchical deep learning model to classify **cryptobenthic fishes** 🐠, a group of small reef fishes. Using transfer learning with various pre-trained models, the aim is to identify the **family, genus, and species** of these fishes with high accuracy while maintaining taxonomic consistency.

## 🚀 Project Overview

In this project, we:

- Use **TensorFlow** and **Keras** to build hierarchical classification models
- Apply **transfer learning** with popular pre-trained models like `ResNet50V2`, `ResNet152V2`, `EfficientNetV2B0/B1`, and `VGG16`
- Implement multiple architectural variants for hierarchical classification:
  - Standard (`std`): Basic hierarchical model
  - Attention-based (`att`): Using taxonomy-conditioned attention
  - Gated (`gated`): Using gated hierarchical fusion
  - Concatenation (`concat`): Using feature concatenation
- Leverage **W&B (Weights & Biases)** for tracking experiments and model artifacts

## 📂 Directory Structure

```bash
cryptovision/
├── cryptovision/
│   ├── __init__.py
│   ├── compare_arch.py      # Architecture comparison script
│   ├── dataset.py           # Dataset loading and preprocessing
│   ├── fine_tuning.py       # Model fine-tuning utilities
│   ├── grid_search.py       # Hyperparameter optimization
│   ├── models.py            # Model architectures
│   ├── settings.yaml        # Configuration file
│   ├── train.py            # Training utilities
│   └── utils.py            # Helper functions
├── models/                  # Saved model weights and artifacts
├── data/                    # Dataset directory
└── README.md               # Project documentation
```

## 🧠 Models & Approach

1. **Hierarchical Classification** 🌳:

   - Multi-level classification (family → genus → species)
   - Ensures taxonomic consistency across levels
   - Uses taxonomy-conditioned attention and gated mechanisms

2. **Transfer Learning** 🌍:

   - Pre-trained models: `ResNet50V2`, `ResNet152V2`, `EfficientNetV2B0/B1`, `VGG16`
   - Fine-tuning capabilities for improved performance

3. **Data Augmentation** 🎨:

   - Random flips, rotations, zooms
   - Contrast and brightness adjustments
   - Gaussian noise
   - Translation and cropping

4. **Architecture Variants** 🏗️:

   - Standard (`std`): Basic hierarchical model
   - Attention (`att`): Uses taxonomy-conditioned attention
   - Gated (`gated`): Implements gated hierarchical fusion
   - Concatenation (`concat`): Uses feature concatenation

5. **W&B Integration** 📊:
   - Experiment tracking
   - Metric visualization
   - Model artifact management
   - Taxonomy alignment monitoring

## 💻 How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/leonardo-reginato/cryptovision.git
   cd cryptovision
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure settings**:
   Edit `cryptovision/settings.yaml` to configure:

   - Model architecture and hyperparameters
   - Data paths and preprocessing
   - Training parameters
   - W&B project settings

4. **Train a model**:

   ```bash
   # Train a single model
   python -m cryptovision.train

   # Compare different architectures
   python -m cryptovision.compare_arch

   # Run hyperparameter optimization
   python -m cryptovision.grid_search
   ```

5. **Fine-tune a model**:
   ```bash
   python -m cryptovision.fine_tuning
   ```

## 📊 Model Architecture Comparison

The project implements and compares four different architectural approaches for hierarchical classification:

1. **Standard (`std`)**:

   - Basic hierarchical model
   - Direct classification at each level

2. **Attention (`att`)**:

   - Uses taxonomy-conditioned attention
   - Modulates features based on higher-level predictions

3. **Gated (`gated`)**:

   - Implements gated hierarchical fusion
   - Controls information flow between levels

4. **Concatenation (`concat`)**:
   - Concatenates features from different levels
   - Combines information for better classification

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x
- Weights & Biases
- PyYAML
- Loguru
- NumPy
- Other dependencies in `requirements.txt`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## 💬 Contact

For questions or suggestions, please:

- Open an issue in this repository
- Contact the maintainers directly

---

Happy coding! 🎣
