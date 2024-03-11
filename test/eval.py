import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import csv
import fcntl

# File path for storing all results
RESULTS_FILE = "ALL-RESULTS.csv"


def lock_file(file):
    """Acquires lock on the given file."""
    fcntl.flock(file, fcntl.LOCK_EX)


def unlock_file(file):
    """Releases lock on the given file."""
    fcntl.flock(file, fcntl.LOCK_UN)


def save_csv(label_results, model_name, epochs, lr):
    """
    Saves the results to a CSV file.

    Args:
        label_results (dict): Dictionary containing label-wise results.
        model_name (str): Name of the model.
        epochs (int): Number of epochs.
        lr (float): Learning rate.

    Returns:
        None
    """
    with open(RESULTS_FILE, 'a', newline='') as f:
        lock_file(f)
        writer = csv.writer(f)
        writer.writerow([model_name, label_results, epochs, lr])
        unlock_file(f)


def _eval(test_data, model, device, labels_available=True):
    """
    Block level function used in _evaluate to perform the metrics calculation.

    Args:
        test_data (Dataset): Test dataset.
        model (nn.Module): Model to evaluate.
        device (torch.device): Device to perform computations on.
        labels_available (bool): Whether labels are available for evaluation (default is True).

    Returns:
        test_preds (Tensor): Predictions made by the model.
        test_labels (Tensor): True labels from the test dataset.
    """
    if labels_available:
        preds = []
        true = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(DataLoader(test_data, batch_size=16)):
                inputs, labels = inputs.to(torch.float32).to(
                    device), labels.to(torch.float32).to(device)
                logps = model.forward(inputs)
                preds.append(logps)
                true.append(labels)
        model.train()
        test_preds = torch.cat(preds).cpu()
        test_labels = torch.cat(true).cpu()

        return test_preds, test_labels
    else:
        preds = []
        model.eval()
        with torch.no_grad():
            for inputs in tqdm(DataLoader(test_data, batch_size=16)):
                inputs = inputs.to(torch.float32).to(device)
                logps = model.forward(inputs)
                preds.append(logps)
        model.train()
        test_preds = torch.cat(preds).cpu()
        return test_preds


def _evaluate_fromckpts(test_dataset=None, ckpt_path=None, device=torch.device("cpu"), epochs=None, lr=None):
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
    assert test_dataset != None, "train dataset object passed is None."
    assert os.path.exists(ckpt_path), str(ckpt_path) + " does not exist."
    assert isinstance(
        device, torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling filetuning function"

    print("Evaluating the model from location: "+ckpt_path)
    model = torch.load(ckpt_path)
    results = _evaluate(test_dataset=test_dataset, model=model,
                        device=device, epochs=epochs, lr=lr)
    return results


def _evaluate(test_dataset=None, model=None, device=torch.device("cpu"), labels_available=True, save_results="./results/", epochs=None, lr=None):
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
    assert test_dataset != None, "train dataset object passed is None."
    assert model != None, "model object passed is None"
    assert isinstance(
        device, torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling filetuning function"

    if labels_available:
        test_preds, test_labels = _eval(
            test_dataset, model, device, labels_available=True)
    else:
        test_preds = _eval(test_dataset, model, device, labels_available=False)

    if labels_available:
        bin_info = str(test_dataset.name.split("-")[-1])
        prediction_type = bin_info
        if prediction_type == "binary":
            results = roc_auc_score(test_labels, test_preds, average=None)
            label_results = _print_results(test_dataset, results)
            save_csv(label_results, model.name, epochs, lr)
        elif prediction_type == "multi":
            results = roc_auc_score(
                test_labels, test_preds, average="weighted", multi_class='ovr')
            print(test_dataset.labels[0], " AUROC: ", round(results, 4))
            label_results = {test_dataset.labels[0]: round(results, 4)}
            save_csv(label_results, model.name, epochs, lr)
        else:
            raise "dataloader component isnt processed through cxrlearn.cxr_to_pt function"
        _save_results(test_dataset, test_preds, test_labels,
                      save_results=save_results, name=model.name+"-")
        return results
    else:
        _save_results(test_dataset, test_preds,
                      save_results=save_results, name=model.name+"-")


def _save_results(test_dataset, test_preds, test_labels=None, save_results="./results/", name=None):
    """
    Save the results of the evaluation to CSV files.

    Args:
        test_dataset (Dataset): Test dataset.
        test_preds (Tensor): Predictions made by the model.
        test_labels (Tensor): True labels from the test dataset (default is None).
        save_results (str): Directory to save results (default is "./results/").
        name (str): Name of the model (default is None).

    Returns:
        None
    """
    if test_labels != None:
        gt = pd.DataFrame(test_labels)
        gt.to_csv(save_results+name+"groundtruth.csv")
    pred = pd.DataFrame(test_preds)
    pred.to_csv(save_results+name+"predictions.csv")

    return


def _print_results(data, results):
    """
    Print the evaluation results.

    Args:
        data (Dataset): Dataset object.
        results (float or ndarray): Evaluation results.

    Returns:
        results_dict (dict): Dictionary containing label-wise results.
    """
    results_dict = {}
    if len(data.labels) > 1:
        for i, label in enumerate(data.labels):
            print(label, " AUROC: ", round(results[i], 4))
            results_dict[label] = round(results[i], 4)
        print("--------------------------------------------")
        print("Average AUROC: ", round(results.mean(), 4))
        results_dict["average"] = round(results.mean(), 4)
    else:
        print(data.labels, " AUROC: ", round(results, 4))
        results_dict["average"] = round(results, 4)
    return results_dict
