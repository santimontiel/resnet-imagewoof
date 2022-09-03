import torch
from torch.utils.data import Dataset

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from typing import Dict

def plot_dataset_batch(dataset: Dataset):
    """
    """
    # @TODO: Move this function from dataset.py to this one.
    pass


def plot_metrics(results: Dict[str, float], path: str, day: str, hour: str) -> None:
    # @TODO: Split subplots to store it.
    # @TODO: Dynamic name to store figures.

    sns.set_theme()

    # Create a mosaic with two figures: losses and metrics.
    fig, axd = plt.subplot_mosaic([
        ["loss", "metrics"]
    ], figsize=(10,5), constrained_layout=True)

    # Losses plot.
    axd["loss"].plot(results["train_loss"], label="Train loss")
    axd["loss"].plot(results["test_loss"], label="Test loss")
    axd["loss"].set_title("Losses", fontsize=12)
    axd["loss"].legend()
    axd["loss"].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Metrics plots.
    axd["metrics"].plot(results["train_acc"], label="Train acc")
    axd["metrics"].plot(results["test_acc"], label="Test acc")
    axd["metrics"].set_title("Accuracy", fontsize=12)
    axd["metrics"].legend()
    axd["metrics"].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Suptitle.
    suptitle = f"ResNet-ImageWoof: Losses and accuracy.\nExperiment: Day {day}, Hour {hour}"
    fig.suptitle(suptitle, fontweight="bold")

    # Save it to results folder.
    fig.savefig(path)

    return fig

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'