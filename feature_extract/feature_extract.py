import torch
# import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


def _get_features(dataset=None, model=None, device=torch.device("cpu"), with_labels=True, max_batches=100000):
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
    all_features = []
    all_labels = []

    c = 0

    if with_labels:
        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=200)):
                if c > max_batches:
                    break
                features = model.forward(images.to(device))
                all_features.append(features)
                all_labels.append(labels)
                c += 1
        return torch.cat(all_features), torch.cat(all_labels)
    else:
        with torch.no_grad():
            for images, _ in tqdm(DataLoader(dataset, batch_size=200)):
                features = model.forward(images.to(device))
                all_features.append(features)

        return torch.cat(all_features), _
