# Torch imports.
import torch
from torch import nn
from torchvision import transforms

# Stdlib imports.
import sys
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime, timedelta

# External imports.
import pandas as pd
from tqdm.auto import tqdm

# Configure relative path.
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Modules imports (once path is solved).
from utils.models import resnet50_imagewoof
from utils.config import config1 as cfg
from utils.engine import train_step, test_step
from utils.dataset import image_woof_train_dataloader, image_woof_test_dataloader
from utils.visualize import Colors as c
from utils.visualize import plot_metrics


def run(model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: str,
        epochs: int = 10,
):

    # Dictionary to keep track of metrics and losses.
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # For every epoch, loop through train and test steps.
    for epoch in tqdm(range(epochs),
                      total=epochs,
                      colour="GREEN",
                      desc=f"{c.BOLD}{c.OKGREEN}Epoch number{c.ENDC}",
                      leave=False,
                      dynamic_ncols=True):

        # First, train step.
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        # Then, test step.
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        # Print the results of this epoch.
        tqdm.write(f"Epoch: {str(epoch).zfill(2)} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        # Update the results tracking dictionary.
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

def main():

    # Set up hardware agnostic code.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set a random seed.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Transformations for train data.
    train_transforms = transforms.Compose([
        transforms.Resize(size=cfg["IMG_SIZE"]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    # Load train and test data.
    train_dataloader = image_woof_train_dataloader(
        path_to_root=Path(cfg["PATH_TO_ROOT"]),
        bs=cfg["BATCH_SIZE"],
        transforms=train_transforms
    )
    test_dataloader = image_woof_test_dataloader(
        path_to_root=Path(cfg["PATH_TO_ROOT"]),
        bs=cfg["BATCH_SIZE"],
        size=cfg["IMG_SIZE"]
    )

    # Load a model from the dispatcher.
    model = resnet50_imagewoof()
    model.to(device=DEVICE)

    # Set up the loss function and the optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=3,
        eta_min=1e-2
    )

    # Let's train!
    start_time = timer()
    results = run(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=cfg["NUM_EPOCHS"],
                device=DEVICE,
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {timedelta(seconds=end_time-start_time)} hours.")

    # Get day and hour information to store results.
    date = datetime.now()
    day = f"{date.year}_{str(date.month).zfill(2)}_{str(date.day).zfill(2)}"
    hour = f"{str(date.hour).zfill(2)}_{str(date.minute).zfill(2)}"

    # Save the metrics and losses as png and csv.
    res_dir = (Path().cwd() / "results" / f"run_{day}_{hour}_train")
    res_dir.mkdir(parents=True, exist_ok=True)
    res_png = plot_metrics(
        results=results,
        path=f"{res_dir}/results.png",
        day=f"{str(date.day).zfill(2)}/{str(date.month).zfill(2)}/{date.year}",
        hour=f"{str(date.hour).zfill(2)}:{str(date.minute).zfill(2)}")
    res_df = pd.DataFrame(results).to_csv(f"{res_dir}/results.csv")

if __name__ == "__main__":
    main()