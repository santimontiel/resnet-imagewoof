# Torch imports.
import torch
from torch import nn

# Stdlib imports.
import sys
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime, timedelta

# External imports.
from tqdm.auto import tqdm

# Configure relative path.
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Repo imports.
from utils.config import config_test_1 as cfg
from utils.dataset import image_woof_test_dataloader
from utils.models import resnet18_imagewoof as model_cls
from utils.engine import test_step
from utils.serialize import serial_val_results

def run(model: torch.nn.Module,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: str):
    
    # Then, test step.
    test_loss, test_acc = test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
    )

    # Print the results.
    tqdm.write(f"Val loss: {test_loss:.4f} | Val acc: {test_acc:.4f}")

    return test_loss, test_acc


def main():
    
    # Set up hardware agnostic code.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set a random seed.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Load test data.
    test_dataloader = image_woof_test_dataloader(
        path_to_root=Path(cfg["PATH_TO_ROOT"]),
        bs=cfg["BATCH_SIZE"],
        size=cfg["IMG_SIZE"]
    )

    # Set up the loss function.
    loss_fn = nn.CrossEntropyLoss()

    # Load the model and the checkpoint.
    model = model_cls().to(DEVICE)
    model_state_dict = (torch.load(cfg["PATH_TO_MODEL"])).get("model_state_dict")
    model.load_state_dict(model_state_dict)

    # Let's validate.
    start_time = timer()
    loss, acc = run(model=model,
                    test_dataloader=test_dataloader,
                    loss_fn=loss_fn,
                    device=DEVICE,
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total validation time: {timedelta(seconds=end_time-start_time)} hours.")
    
    # Get day and hour information to store results.
    date = datetime.now()
    day = f"{date.year}_{str(date.month).zfill(2)}_{str(date.day).zfill(2)}"
    hour = f"{str(date.hour).zfill(2)}_{str(date.minute).zfill(2)}"

    # Save the metrics and losses as png and csv.
    res_dir = (Path().cwd() / "runs" / f"{day}_{hour}_val" / "results")
    res_dir.mkdir(parents=True, exist_ok=True)
    serial_val_results(res_dir, loss, acc)


if __name__ == "__main__":
    main()