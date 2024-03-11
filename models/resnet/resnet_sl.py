import torch
from torchvision import models, transforms

def _resnet(device=torch.device("cpu"), num_out=1):
    """
    Load the ResNet model.

    Args:
        device (torch.device): Device to load the model on (default is CPU).
        num_out (int): Number of output classes (default is 1).

    Returns:
        torch.nn.Module: Loaded ResNet model.
    """
    model = models.resnet50(pretrained=True).to(device)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_out)
    model.fc = model.fc.to(device)

    model.name = "resnet"

    return model
