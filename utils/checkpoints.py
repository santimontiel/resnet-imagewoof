import torch
from pathlib import Path

class BestAndLastCheckpoints:

    def __init__(self, model, optimizer) -> None:
        """ Class to save the best and last model checkpoints while training.
        """  
        self.last_info = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float('inf')
        }
        self.best_info = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float('inf')
        }

    def step(self, epoch, this_loss, model, optimizer) -> None:
        self.last_info = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': this_loss
        }
        if this_loss <= self.best_info['loss']:
            self.best_info = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': this_loss
            }

    def save(self, path: Path) -> None:
        torch.save(self.last_info, f"{path}/last.pt")
        torch.save(self.best_info, f"{path}/best.pt")
