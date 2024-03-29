import torch
import warnings
from .utils.model_se import densenet121


def _s2mts2(device=torch.device("cpu"), freeze_backbone=True, num_out=None):
    """
    S2MTS2 - Self-supervised Mean Teacher for Semi-supervised Chest X-ray Classification https://arxiv.org/pdf/2103.03629.pdf
    Load the S2MTS2 model. Implementation Reference - 

    Args:
        device (torch.device): Device to load the model on (default is CPU).
        freeze_backbone (bool): Whether to freeze the backbone layers of the model (default is True).
        num_out (int): Number of output classes (default is None).

    Returns:
        torch.nn.Module: Loaded S2MTS2 model.
    """
    if device == torch.device("cpu"):
        warnings.warn("Loading model on CPU.... Use GPU if available for faster training! pass device variable in s2mts2 function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    assert isinstance(
        freeze_backbone, bool), "freeze_backbone can only be a bool (True/False) value"
    assert isinstance(
        device, torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling s2mts2 function"

    model = densenet121(pretrained=True, progress=True)

    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, in_features)

    ckpt = torch.load("./pretrained/s2mts2-chexpert.pth.tar")

    state_dict = ckpt["state_dict"]

    for k in list(state_dict.keys()):
        if k.startswith("encoder_q") and not k.startswith("encoder_q.{}".format("classifier_group")):
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    if num_out == None:
        model.name = "s2mts2-logreg"
        return model.to(device)

    model.classifier_group = torch.nn.Sequential(
        torch.nn.Linear(in_features, in_features),
        torch.nn.LeakyReLU(0.1, inplace=True),
        torch.nn.Linear(in_features, num_out, bias=False))

    if freeze_backbone:
        for name, param in list(model.named_parameters())[:-3]:
            param.requires_grad = False
    if freeze_backbone:
        model.name = "s2mts2-linear"
    else:
        model.name = "s2mts2-finetune"
    return model.to(device)
