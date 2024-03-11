from . import models
import torch
import copy
import warnings
from .models.modeling import VisionTransformer, CONFIGS, Transformer


def _load_weights(model, weight_path, device):
    """
    Load pre-trained weights into the provided model.

    Args:
        model (torch.nn.Module): Model to load weights into.
        weight_path (str): Path to the pre-trained weights file.
        device (torch.device): Device to load the weights on.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    pretrained_weights = torch.load(weight_path, map_location=device)
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items()
                    if k in model_weights}

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    return model


def _setup_REFERS(img_size, num_classes, pretrained_dir, device, get_fmap=False):
    """
    Set up the REFERS model.

    Args:
        img_size (int): Size of the input image.
        num_classes (int): Number of output classes.
        pretrained_dir (str): Directory containing the pre-trained weights.
        device (torch.device): Device to load the model on.
        get_fmap (bool): Whether to get feature maps (default is False).

    Returns:
        torch.nn.Module: Initialized REFERS model.
    """
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(
        config, img_size, zero_head=True, num_classes=num_classes, get_fmap=get_fmap)
    model = _load_weights(model, pretrained_dir, device)
    model.to(device)

    return model


def _setup_transformer(img_size, pretrained_dir, device):
    """
    Set up the Transformer model.

    Args:
        img_size (int): Size of the input image.
        pretrained_dir (str): Directory containing the pre-trained weights.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Initialized Transformer model.
    """
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, img_size, zero_head=True, get_fmap=True)
    model = _load_weights(model, pretrained_dir, device)

    model.to(device)

    return model


class _PretrainedREFER(torch.nn.Module):
    def __init__(
        self,
        model_in: torch.nn.Module,
        freeze_encoder: bool = True,
    ):
        super(_PretrainedREFER, self).__init__()
        self.model_in = model_in
        if freeze_encoder:
            for param in list(self.model_in.parameters())[:-2]:
                param.requires_grad = False

    def forward(self, x):
        pred = self.model_in(x)
        return pred


def _refers(device=torch.device("cpu"), freeze_backbone=True, num_out=None):
    """
    REFERS - Generalized radiograph representation learning via cross-supervision between images and free-text radiology reports https://www.nature.com/articles/s42256-021-00425-9.pdf
    Load the REFERS model.

    Args:
        device (torch.device): Device to load the model on (default is CPU).
        freeze_backbone (bool): Whether to freeze the backbone layers (default is True).
        num_out (int): Number of output classes (default is None).

    Returns:
        torch.nn.Module: Loaded REFERS model.
    """
    if device == torch.device("cpu"):
        warnings.warn("Loading model on CPU.... Use GPU if available for faster training! pass device variable in refers function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    # assert num_out>0, "num_out (Number of classes output) has to be greater than 0!"
    assert isinstance(
        freeze_backbone, bool), "freeze_backbone can only be a bool (True/False) value"
    assert isinstance(
        device, torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling refers function"

    if num_out == None:
        model = _setup_transformer(
            img_size=224, pretrained_dir="./pretrained/refers_checkpoint.pth", device=device)
        model.name = "refers-logreg"
        return model.to(device)

    refers_model = _setup_REFERS(img_size=224, num_classes=num_out,
                                 pretrained_dir="./pretrained/refers_checkpoint.pth", device=device)
    model = _PretrainedREFER(refers_model, freeze_backbone)
    del refers_model

    if num_out == None:
        model.name = "refers-logreg"
        return model.to(device)

    if freeze_backbone:
        model.name = "refers-linear"
    else:
        model.name = "refers-finetune"

    return model.to(device)
