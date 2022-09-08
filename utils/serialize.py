import json
from pathlib import Path

def serial_val_results(path: Path, loss: float, acc: float):
    """ After finishing validation script, log final loss and accuracy
    to a JSON file in the runs folder.
    
    Args:
        - path [Path]
            Path to the results folder.
        - loss [float]
            Final loss after validation step.
        - acc [float]
            Final accuracy after validation step.
    """
    data_dict = {
        "val_loss": loss,
        "val_acc": acc,
    }

    with open(path / "results.json", "w") as json_file:
        json.dump(data_dict, json_file)

def serial_train_state():
    """ @TODO: Serialize model state dict, optim state dict and
            scheduler to JSON file. Research also how to obtain
            optimizer type.
    """
    pass