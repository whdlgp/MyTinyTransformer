import os
import torch


def save_checkpoint(path, model, optimizer, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    torch.save(checkpoint, path)

    root, ext = os.path.splitext(path)
    torch.save(checkpoint, f"{root}_epoch{epoch}{ext}")


def load_checkpoint(path, model, device, optimizer=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"]
