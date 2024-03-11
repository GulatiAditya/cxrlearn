import torch
from torchvision import models, transforms
import warnings


def _convirt(device=torch.device("cpu"), freeze_backbone=True, num_out=None):
    """
    https://arxiv.org/pdf/2010.00747.pdf ConVIRT - CONTRASTIVE LEARNING OF MEDICAL VISUAL REPRESENTATIONS FROM PAIRED IMAGES AND TEXT 
    Loads the ConVirt model based on the ResNet50 architecture.

    Args:
        device (torch.device): The device to load the model on. Defaults to CPU.
        freeze_backbone (bool): Whether to freeze the backbone layers of the loaded model. If True, only the last fully connected layer
                                is trained. Defaults to True.
        num_out (int): Number of output classes for the model. If None, the model is loaded without the final fully connected layer,
                       suitable for feature extraction or transfer learning. Defaults to None.

    Returns:
        torch.nn.Module: The loaded ConVirt model.

    Raises:
        AssertionError: If the freeze_backbone argument is not a boolean.
        AssertionError: If the device argument is not a torch.device object.
    """
    if device == torch.device("cpu"):
        warnings.warn("Loading model on CPU.... Use GPU if available for faster training! pass device variable in convirt function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")

    assert isinstance(
        freeze_backbone, bool), "freeze_backbone can only be a bool (True/False) value"
    assert isinstance(
        device, torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling convirt function"

    PATH = './pretrained/convirt_chest_mimic.pt'

    model = models.resnet50(pretrained=True).to(device)
    model.load_state_dict(torch.load(PATH, map_location=device), strict=False)

    if num_out == None:
        model.name = "convirt-logreg"
        return model.to(device)

    # print(k0)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential()

    model.fc = torch.nn.Linear(num_ftrs, num_out)
    model.fc = model.fc.to(device)

    if freeze_backbone:
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False

    if freeze_backbone:
        model.name = "convirt-linear"
    else:
        model.name = "convirt-finetune"

    return model.to(device)
