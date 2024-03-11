# Import necessary modules
from gc import freeze
from datasets.prepare_dataset import _cxr_to_pt

# Import functions from custom modules
from models.gloria.gloria_cxrt import _gloria
from models.refers.refers_cxrt import _refers
from models.s2mts2.s2mts2_cxrt import _s2mts2
from models.convirt.convirt_cxrt import _convirt
from models.chexzero.chexzero_cxrt import _chexzero
from models.medaug.medaug_cxrt import _medaug
from models.mococxr.mococxr_cxrt import _mococxr
from models.resnet.resnet_sl import _resnet

from feature_extract.feature_extract import _get_features

from train.finetuning import _finetune
from test.eval import _evaluate, _evaluate_fromckpts

from train.finetuning import _finetune_logreg

# Import torch and torch optimizer
import torch
from torch import optim

# Function to convert dataset to PyTorch format


def cxr_to_pt(csv_file, path_col, class_col, dataset_name, out_pt_address, reshape_size=(224, 224), pt_withoutLabels=False, prediction_type="binary", skip_loading=False, fewshot_class=None, fewshot_perclass=0, save_data=True):
    """
    Convert a CSV file containing chest X-ray data to a PyTorch .pt file.

    Args:
        csv_file (str): Path to the CSV file containing the data.
        path_col (list): List of image path columns.
        class_col (list): List of target class columns.
        dataset_name (str): Name of the dataset.
        out_pt_address (str): Output directory where the .pt file will be saved.
        reshape_size (tuple): Shape to which images are resized (default is (224, 224)).
        pt_withoutLabels (bool): Flag indicating if the dataset contains labels.
        prediction_type (str): Type of prediction task ("binary" or "multi").
        skip_loading (bool): Whether to skip loading if a .pt file already exists.
        fewshot_class (str): Column indicating the class to perform few-shot learning on.
        fewshot_perclass (int): Number of samples per class for few-shot learning.
        save_data (bool): Whether to save the dataset.

    Returns:
        _CXR_Dataset: Converted dataset.
    """
    return _cxr_to_pt(csv_file, path_col, class_col, dataset_name, out_pt_address, reshape_size=reshape_size, pt_withoutLabels=pt_withoutLabels, prediction_type=prediction_type, skip_loading=skip_loading, fewshot_class=fewshot_class, fewshot_perclass=fewshot_perclass, save_data=save_data)

# Functions to initialize different models


def chexzero(device=torch.device("cpu"), freeze_backbone=True, linear_layer_dim=512, num_out=None):
    """
    https://proceedings.mlr.press/v158/endo21a.html Retrieval-Based Chest X-Ray Report Generation Using a Pre-trained Contrastive Language-Image Model
    Loads the CheXZero model with the CLIP architecture.

    Args:
        device (torch.device): The device to load the model on. Defaults to CPU.
        freeze_backbone (bool): Whether to freeze the backbone layers of the loaded model. If True, only the last linear layer
                                is trained. Defaults to True.
        linear_layer_dim (int): Dimension of the linear layer added on top of the CLIP model. Defaults to 512.
        num_out (int): Number of output classes for the model. If None, the model is loaded without the final linear layer,
                       suitable for feature extraction or transfer learning. Defaults to None.

    Returns:
        torch.nn.Module: The loaded CheXZero model with the CLIP architecture.

    Raises:
        AssertionError: If the linear_layer_dim argument is not greater than 0.
        AssertionError: If the freeze_backbone argument is not a boolean.
        AssertionError: If the device argument is not a torch.device object.
    """
    return _chexzero(device=device, freeze_backbone=freeze_backbone, linear_layer_dim=linear_layer_dim, num_out=num_out)


def convirt(device=torch.device("cpu"), freeze_backbone=True, num_out=None):
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
    return _convirt(device=device, freeze_backbone=freeze_backbone, num_out=num_out)


def gloria(model="resnet50", device=torch.device("cpu"), freeze_backbone=True, num_ftrs=2048, num_out=None):
    """
    https://openaccess.thecvf.com/content/ICCV2021/papers/ GLoRIA - GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition 
    Loads a GLoRIA (GLobal interpretable explORAtion) model based on the ResNet architecture.

    Args:
        model (str): The model architecture to use. Supported options are "resnet50" and "resnet18". Defaults to "resnet50".
        device (torch.device): The device to load the model on. Defaults to CPU.
        freeze_backbone (bool): Whether to freeze the backbone layers of the loaded model. If True, only the last fully connected layer
                                is trained. Defaults to True.
        num_ftrs (int): Number of features in the last layer of the loaded model. Defaults to 2048.
        num_out (int): Number of output classes for the model. If None, the model is loaded without the final fully connected layer,
                       suitable for feature extraction or transfer learning. Defaults to None.

    Returns:
        torch.nn.Module: The loaded GLoRIA model.

    Raises:
        AssertionError: If an unsupported model architecture is specified.
        AssertionError: If the num_ftrs argument is not greater than 0.
        AssertionError: If the freeze_backbone argument is not a boolean.
        AssertionError: If the device argument is not a torch.device object.
    """
    return _gloria(model=model, device=device, freeze_backbone=freeze_backbone, num_ftrs=num_ftrs, num_out=num_out)


def medaug(model="resnet50", pretrained_on="mimic-cxr", device=torch.device("cpu"), freeze_backbone=True, num_out=None):
    """
    https://arxiv.org/abs/2102.10663 MedAug: Contrastive learning leveraging patient metadata improves representations for chest X-ray interpretation 
    Loads a medical image augmentation (MedAug) model based on the ResNet architecture.

    Args:
        model (str): The model architecture to use. Supported options are "resnet50" and "resnet18".
                     Defaults to "resnet50".
        pretrained_on (str): The dataset the model was pretrained on. Supported options are "mimic-cxr" and "chexpert".
                             Defaults to "mimic-cxr".
        device (torch.device): The device to load the model on. Defaults to CPU.
        freeze_backbone (bool): Whether to freeze the backbone layers of the loaded model. If True, only the last fully connected layer
                                is trained. Defaults to True.
        num_out (int): Number of output classes for the model. If None, the model is loaded without the final fully connected layer,
                       suitable for feature extraction or transfer learning. Defaults to None.

    Returns:
        torch.nn.Module: The loaded MedAug model.

    Raises:
        AssertionError: If an unsupported model architecture or dataset is specified.
        AssertionError: If the freeze_backbone argument is not a boolean.
        AssertionError: If the device argument is not a torch.device object.
    """
    return _medaug(model=model, pretrained_on=pretrained_on, device=device, freeze_backbone=freeze_backbone, num_out=num_out)


def mococxr(model="resnet50", device=torch.device("cpu"), freeze_backbone=True, num_out=None):
    """
    https://arxiv.org/pdf/2010.05352.pdf MocoCXR - MoCo-CXR: MoCo Pretraining Improves Representation and Transferability of Chest X-ray Models.
    Load the MOCO-CXR model.

    Args:
        model (str): Model architecture to use (default is 'resnet50').
        device (torch.device): Device to load the model on (default is CPU).
        freeze_backbone (bool): Whether to freeze the backbone layers (default is True).
        num_out (int): Number of output classes (default is None).

    Returns:
        torch.nn.Module: Loaded MOCO-CXR model.
    """
    return _mococxr(model=model, device=device, freeze_backbone=freeze_backbone, num_out=num_out)


def refers(device=torch.device("cpu"), freeze_backbone=True, num_out=None):
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
    return _refers(device=device, freeze_backbone=freeze_backbone, num_out=num_out)


def s2mts2(device=torch.device("cpu"), freeze_backbone=True, num_out=None):
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
    return _s2mts2(device=device, freeze_backbone=freeze_backbone, num_out=num_out)


def resnet(device=torch.device("cpu"), num_out=1):
    """
    Load the ResNet model.

    Args:
        device (torch.device): Device to load the model on (default is CPU).
        num_out (int): Number of output classes (default is 1).

    Returns:
        torch.nn.Module: Loaded ResNet model.
    """
    return _resnet(device=device, num_out=num_out)

# Function to extract features


def get_features(dataset=None, model=None, device=torch.device("cpu"), with_labels=True, max_batches=100000):
    """
    Extract features from a given dataset using a model.

    Args:
        dataset (Dataset): Dataset containing the images.
        model (nn.Module): Pre-trained model for feature extraction.
        device (torch.device): Device to perform computation (default is torch.device("cpu")).
        with_labels (bool): Flag indicating whether the dataset contains labels (default is True).
        max_batches (int): Maximum number of batches to process (default is 100000).

    Returns:
        tuple: Tuple containing the extracted features and labels (if available).
    """
    return _get_features(dataset=dataset, model=model, device=device, with_labels=with_labels, max_batches=max_batches)

# Function for fine-tuning a model


def finetune(train_dataset=None, val_dataset=None, model=None, device=torch.device("cpu"), optimizer=optim.SGD, scheduler=optim.lr_scheduler.CosineAnnealingLR, scheduler_stepping=200, batch_size=32, epochs=100, lr=0.001, momentum=0.9, shuffle=True, num_workers=4, ckpt_path="./pt-finetune"):
    """
    Fine-tune the whole model from the pretrained checkpoints. All layers in the model are tunable.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        model (nn.Module): Model to fine-tune.
        device (torch.device): Device to perform computations on (default is CPU).
        optimizer (torch.optim): Optimizer for training (default is SGD).
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler (default is CosineAnnealingLR).
        scheduler_stepping (int): Number of steps before stepping the scheduler (default is 200).
        batch_size (int): Batch size for training (default is 32).
        epochs (int): Number of epochs (default is 100).
        lr (float): Learning rate (default is 0.001).
        momentum (float): Momentum factor (default is 0.9).
        shuffle (bool): Whether to shuffle the data during training (default is True).
        num_workers (int): Number of worker processes for data loading (default is 4).
        ckpt_path (str): Path to save model checkpoints (default is "./pt-finetune").

    Raises:
        AssertionError: If any of the input arguments are invalid or None.

    Returns:
        None
    """
    return _finetune(train_dataset=train_dataset, val_dataset=val_dataset, model=model, device=device, optimizer=optimizer, scheduler=scheduler, scheduler_stepping=scheduler_stepping, batch_size=batch_size, epochs=epochs, lr=lr, momentum=momentum, shuffle=shuffle, num_workers=num_workers, ckpt_path=ckpt_path)

# Function for fine-tuning a logistic regression model


def finetune_logreg(train_dataset=None, test_dataset=None, model=None, train_max_batches=None, test_max_batches=None, device=torch.device("cpu"), random_state=0, C=0.0327, max_iter=500, verbose=0):
    """
    Finetune a logistic regression level layer on top of the Pretrained Model. All weights are frozen except the last logistic regression layer.

    Args:
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Test dataset.
        model (nn.Module): Model for feature extraction.
        train_max_batches (int): Maximum number of batches for training (default is None).
        test_max_batches (int): Maximum number of batches for testing (default is None).
        device (torch.device): Device to perform computations on (default is CPU).
        random_state (int): Random state for logistic regression (default is 0).
        C (float): Inverse of regularization strength (default is 0.0327).
        max_iter (int): Maximum number of iterations (default is 500).
        verbose (int): Verbosity mode (default is 0).

    Returns:
        array_like: Evaluation results.
    """
    return _finetune_logreg(train_dataset=train_dataset, test_dataset=test_dataset, model=model, train_max_batches=train_max_batches, test_max_batches=test_max_batches, device=device, random_state=random_state, C=C, max_iter=max_iter, verbose=verbose)

# Function to evaluate a model


def evaluate(test_dataset=None, model=None, device=torch.device("cpu"), epochs=None, lr=None):
    """
    Evaluate the model saved at the last epoch of the whole training.

    Args:
        test_dataset (Dataset): Test dataset.
        model (nn.Module): Model to evaluate.
        device (torch.device): Device to perform computations on.
        labels_available (bool): Whether labels are available for evaluation (default is True).
        save_results (str): Directory to save results (default is "./results/").
        epochs (int): Number of epochs (unused).
        lr (float): Learning rate (unused).

    Returns:
        results (float or dict): AUC-ROC score or label-wise AUC-ROC scores of the evaluation.
    """
    return _evaluate(test_dataset=test_dataset, model=model, device=device, epochs=epochs, lr=lr)

# Function to evaluate a model using checkpoints


def evaluate_fromckpts(test_dataset=None, ckpt_path=None, device=torch.device("cpu"), epochs=None, lr=None):
    """
    Evaluate the model from checkpoints.

    Args:
        test_dataset (Dataset): Test dataset.
        ckpt_path (str): Path to the checkpoint file.
        device (torch.device): Device to perform computations on.
        epochs (int): Number of epochs (unused).
        lr (float): Learning rate (unused).

    Returns:
        results (float): AUC-ROC score of the evaluation.
    """
    return _evaluate_fromckpts(test_dataset=test_dataset, ckpt_path=ckpt_path, device=device, epochs=epochs, lr=lr)
