import torch
import numpy as np
from torchmetrics.functional.image import structural_similarity_index_measure as lightning_ssim


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    return device


def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def psnr(pred, target):
    # psnr metric
    return 10 * torch.log10(1 / mse_loss(pred, target))


def ssim(pred, target):
    # ssim metric
    return lightning_ssim(pred,target)


def apply_metric(pred, target, metric):
    return metric(pred, target)


def save_model(epoch, model, optimizer, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path
    )


def load_model(model, path):
    model.load_state_dict(torch.load(path))

    return model


def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer
