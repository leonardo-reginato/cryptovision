
# 🐟 CryptoVision - Pretrained Model Selection

Welcome to **CryptoVision**! This project focuses on building a deep learning model to classify **cryptobenthic fishes** 🐠, a group of small reef fishes. Using transfer learning with various pre-trained models, the aim is to identify the **family, genus, and species** of these fishes with high accuracy.

## 🚀 Project Overview

In this project, we:
- Use **TensorFlow** and **Keras** to build and fine-tune models for image classification.
- Apply **transfer learning** with popular pre-trained models like `VGG16`, `ResNet50V2`, `MobileNetV2`, `EfficientNetV2B0`, and more.
- Optimize each model using **fine-tuning techniques** to achieve the best accuracy.
- Leverage **W&B (Weights & Biases)** for tracking experiments, logging metrics, and managing model artifacts.

## 📂 Directory Structure

```bash
.
├── data/
│   ├── processed/
│   │   ├── train/       # Training images
│   │   ├── valid/       # Validation images
│   │   └── test/        # Testing images
├── models/              # Saved models and artifacts
├── notebooks/           # Jupyter notebooks for experiments
├── scripts/             # Python scripts for training and evaluation
└── README.md            # Project documentation
```

## 🧠 Models & Approach

1. **Transfer Learning** 🌍: We use pre-trained models like `VGG16`, `ResNet50V2`, `MobileNetV2`, etc., to leverage existing feature extraction capabilities.
2. **Fine-Tuning** 🔧: Each model is further tuned on our fish dataset to improve performance.
3. **Data Augmentation** 🎨: Includes random flips, rotations, zooms, and other transformations to make our model more robust.
4. **W&B Integration** 📊: Track training progress, visualize results, and store models as artifacts.

## 💻 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/cryptovision.git
   cd cryptovision
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ installed. Then, run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   Organize your dataset in the `data/processed` folder with subfolders for `train`, `valid`, and `test`.

4. **Train a model**:
   Use the provided script to train models:
   ```bash
   python scripts/train_model.py
   ```

5. **Track your experiments**:
   Make sure to have a W&B account and set up the project to track training:
   ```bash
   wandb login
   ```

6. **Evaluate a model**:
   After training, evaluate the performance on the test set:
   ```bash
   python scripts/evaluate_model.py
   ```

## 📊 Visualization & Results

All model training runs and evaluations are tracked with **W&B**. You can check out our project dashboard here: [W&B Project Link](https://wandb.ai/yourusername/cryptovision).

### Models Evaluated:
- `VGG16`
- `ResNet50V2`
- `MobileNetV2`
- `EfficientNetV2B0`

We compare these models to identify the best one for classifying cryptobenthic fishes 🐠.

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x
- W&B
- Matplotlib
- Scikit-learn
- Other dependencies in `requirements.txt`

## 🤝 Contributing

Contributions are welcome! If you want to add improvements, feel free to submit a pull request. For major changes, please open an issue to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## 💬 Feedback

Feel free to reach out if you have any questions or suggestions! You can create an issue in this repo, or contact me directly via [your email](mailto:youremail@example.com).

---

Happy coding! 🎣
