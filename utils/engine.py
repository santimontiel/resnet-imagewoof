import torch
from tqdm.auto import tqdm

# @TODO: Training engine.

def get_lr_from_optimizer(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr_from_scheduler(scheduler: torch.optim.lr_scheduler._LRScheduler) -> float:
    return scheduler.get_last_lr()[0]


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               device: str):

    # Put the model in train mode.
    model.train()

    # Loss and accuracy values for training.
    train_loss, train_acc = 0, 0

    # Loop through every batch of data in data loader.
    batch_len = len(dataloader)

    for batch, (x, y) in tqdm(enumerate(dataloader),
                              total=len(dataloader),
                              desc="Train step",
                              colour="blue",
                              dynamic_ncols=True,
                              leave=False):

        # Send data to target device.
        x, y = x.to(device), y.to(device)

        # Zero grad the optimizer.
        optimizer.zero_grad()

        # Forward pass.
        y_pred = model(x)

        # Calculate loss.
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backward pass.
        loss.backward()

        # Step the optimizer.
        optimizer.step()

        # Calculate accuracy metric.
        y_pred_cls = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_cls == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch.
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    # Step the learning rate scheduler.
    if scheduler is not None:
        scheduler.step()
        eta = get_lr_from_scheduler(scheduler)
    else:
        eta = get_lr_from_optimizer(optimizer)

    return train_loss, train_acc, eta


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str):

    # Put the model in eval mode.
    model.eval()

    # Loss and accuracy values for test.
    test_loss, test_acc = 0, 0

    # Open the no_grad (inference mode) context.
    with torch.inference_mode():

        # Loop through batches of data in data loader.
        for batch, (x, y) in tqdm(enumerate(dataloader),
                                  total=len(dataloader),
                                  desc="Test step",
                                  colour="yellow",
                                  dynamic_ncols=True,
                                  leave=False):

            # Send data to target device.
            x, y = x.to(device), y.to(device)

            # Forward pass.
            y_pred = model(x)

            # Calculate loss.
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Calculate accuracy.
            y_pred_cls = y_pred.argmax(dim=1)
            test_acc += ((y_pred_cls == y).sum().item() / len(y_pred_cls))

    # Adjust metrics to get average loss and accuracy per batch.
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

