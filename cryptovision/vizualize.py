import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def plot_training_history(results, save_path):
    results_df = pd.DataFrame(results.history)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    ax = ax.flatten()

    ax[0].plot(results_df["accuracy"], label="Training Accuracy")
    ax[0].plot(results_df["val_accuracy"], label="Validation Accuracy")
    ax[0].set_title("Accuracy")
    ax[0].legend()

    ax[1].plot(results_df["loss"], label="Training Loss")
    ax[1].plot(results_df["val_loss"], label="Validation Loss")
    ax[1].set_title("Loss")
    ax[1].legend()

    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    plt.figure(figsize=(15, 15))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
    )
    plt.xticks(
        ticks=np.arange(len(labels)), labels=labels.values(), rotation=45, ha="right"
    )
    plt.yticks(
        ticks=np.arange(len(labels)), labels=labels.values(), rotation=0, va="center"
    )
    plt.savefig(save_path)
    plt.close()