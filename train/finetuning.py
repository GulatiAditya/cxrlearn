import torch
import os
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch import optim
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import warnings


def _get_features(dataset=None, model=None, device=torch.device("cpu"), with_labels=True, max_batches=100000):
    """
    Extract features from a given dataset using a specified model.

    Args:
        dataset (Dataset): Dataset to extract features from.
        model (nn.Module): Model to use for feature extraction.
        device (torch.device): Device to perform computations on (default is CPU).
        with_labels (bool): Whether to include labels in the feature extraction (default is True).
        max_batches (int): Maximum number of batches to process (default is 100000).

    Returns:
        torch.Tensor: Extracted features.
        torch.Tensor: Labels (if with_labels=True).

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


def _print_results(data, results):
    """
    Print evaluation results.

    Args:
        data (Dataset): Dataset used for evaluation.
        results (array_like): Evaluation results.
    """
    if len(data.labels) > 1:
        for i, label in enumerate(data.labels):
            print(label, " AUROC: ", round(results[i], 4))
        print("--------------------------------------------")
        print("Average AUROC: ", round(results.mean(), 4))
    else:
        print(data.labels, " AUROC: ", round(results, 4))


def _save_results(test_dataset, test_preds, test_labels=None, save_results="./results/", name=None):
    """
    Save evaluation results to CSV files.

    Args:
        test_dataset (Dataset): Test dataset.
        test_preds (array_like): Predicted labels.
        test_labels (array_like): Ground truth labels.
        save_results (str): Directory to save results (default is "./results/").
        name (str): Name prefix for saved files.
    """
    if test_labels != None:
        gt = pd.DataFrame(test_labels)
        gt.to_csv(save_results+name+"groundtruth.csv")
    pred = pd.DataFrame(test_preds)
    pred.to_csv(save_results+name+"predictions.csv")

    return


def _finetune_logreg(train_dataset=None, test_dataset=None, model=None, train_max_batches=None, test_max_batches=None, device=torch.device("cpu"), random_state=0, C=0.0327, max_iter=500, verbose=0):
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
    bin_info = str(train_dataset.name.split("-")[-1])
    prediction_type = bin_info

    if train_max_batches != None:
        train_x, train_y = _get_features(
            dataset=train_dataset, model=model, device=device, max_batches=train_max_batches)
    else:
        train_x, train_y = _get_features(
            dataset=train_dataset, model=model, device=device)

    if test_max_batches != None:
        test_x, test_y = _get_features(
            dataset=test_dataset, model=model, device=device, max_batches=test_max_batches)
    else:
        test_x, test_y = _get_features(
            dataset=test_dataset, model=model, device=device)

    if prediction_type == "binary":
        class_cols = list(test_dataset.labels)

        pred_y = []
        for i in range(len(class_cols)):
            classifier = LogisticRegression(
                random_state=random_state, C=C, max_iter=max_iter, verbose=verbose)
            classifier.fit(train_x.cpu(), train_y[:, i].cpu())
            test_pred = classifier.predict_proba(test_x.cpu())
            pred_y.append(test_pred[:, 1])

        pred_y = np.transpose(np.asarray(pred_y))
        # print("in function")
        # print(pred_y)

        results = roc_auc_score(test_y.numpy(), pred_y, average=None)
        _print_results(test_dataset, results)
        _save_results(test_dataset, pred_y, test_labels=test_y.cpu(),
                      save_results="./results/", name=model.name+"-")
        return results
    elif prediction_type == "multi":
        print("--------Multi----------")

        test_labels = np.where(train_y == 1)[1]

        _, class_cols = train_y.size()
        classifier = LogisticRegression(
            random_state=0, C=0.316, max_iter=100, verbose=0, multi_class='multinomial')
        classifier.fit(train_x.cpu(), test_labels)

        pred_y = classifier.predict_proba(test_x.cpu())

        results = roc_auc_score(
            test_y, pred_y, multi_class='ovr', average="weighted")
        print(" AUROC: ", round(results, 4))
        _save_results(test_dataset, pred_y, test_labels=test_y.cpu(),
                      save_results="./results/", name=model.name+"-")


def _finetune(train_dataset=None, val_dataset=None, model=None, device=torch.device("cpu"), optimizer=optim.SGD, scheduler=optim.lr_scheduler.CosineAnnealingLR, scheduler_stepping=200, batch_size=32, epochs=100, lr=0.001, momentum=0.9, shuffle=True, num_workers=4, ckpt_path="./pt-finetune"):
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
    if device == torch.device("cpu"):
        warnings.warn("Finetuning on CPU.... Use GPU if available for faster training! pass device variable in chexzero function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    if val_dataset == None:
        warnings.warn(
            "Validation set is None, it is recommended to use a validation set!")

    assert train_dataset != None, "train dataset object passed is None."
    assert model != None, "model object passed is None"
    assert isinstance(
        device, torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling filetuning function"
    assert scheduler_stepping > 0, "scheduler_stepping should be greater than 0!"
    assert batch_size > 0, "batch_size should be greater than 0!"
    assert epochs > 0, "epochs should be greater than 0!"
    assert lr > 0, "lr should be greater than 0!"
    assert momentum > 0, "momentum should be greater than 0!"
    assert num_workers > 0, "num_workers should be greater than 0!"
    assert isinstance(
        shuffle, bool), "shuffle should be a True or False boolean"
    assert isinstance(ckpt_path, str), "ckpt_path has to be a string"

    if not os.path.exists(ckpt_path):
        print(str(ckpt_path) + " does not exist. Creating the path....")
        os.mkdir(str(ckpt_path))

    bin_info = str(train_dataset.name.split("-")[-1])
    prediction_type = bin_info

    if prediction_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif prediction_type == "multi":
        criterion = nn.CrossEntropyLoss()
    else:
        raise "dataloader component isnt processed through cxrlearn.cxr_to_pt function"

    optimizer = optimizer(model.parameters(), lr=lr, momentum=momentum)
    scheduler = scheduler(optimizer, scheduler_stepping)

    model.train()

    steps = 0
    batch = 0

    running_loss = 0
    best_val_loss = 1000
    best_val_auc = 0
    best_epoch = 0
    early_stop_epochs = 15
    train_losses, val_losses, val_aucs = [], [], []

    print("-------------------------------------")
    print("Training is starting..")
    print("-------------------------------------")
    for epoch in range(epochs):
        steps = 0
        for inputs, labels in tqdm(DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)):
            steps += 1
            inputs, labels = inputs.to(torch.float32).to(
                device), labels.to(torch.float32).to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch += labels.size()[0]
            # if steps % epochs/4 == 0:
            #     scheduler.step()

        if val_dataset != None:
            val_loss = 0
            accuracy = 0
            model.eval()
            preds_v = []
            true_v = []
            with torch.no_grad():
                for inputs_val, labels_val in tqdm(DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)):
                    inputs_val, labels_val = inputs_val.to(torch.float32).to(
                        device), labels_val.to(torch.float32).to(device)
                    logps = model.forward(inputs_val)
                    batch_loss = criterion(logps, labels_val)
                    val_loss += batch_loss.item()
                    preds_v.append(logps)
                    true_v.append(labels_val)
            val_preds = torch.cat(preds_v).cpu()
            val_labels = torch.cat(true_v).cpu()

            if prediction_type == "binary":
                results = roc_auc_score(
                    val_labels, val_preds, average="weighted")
            elif prediction_type == "multi":
                results = roc_auc_score(
                    val_labels, val_preds, average="weighted", multi_class='ovr')

            train_losses.append(running_loss/batch)
            val_losses.append(val_loss/len(val_dataset))
            val_aucs.append(results.mean())

            if val_loss/len(val_dataset) < best_val_loss:
                best_val_loss = val_loss/len(val_dataset)
            if results.mean() > best_val_auc:
                best_val_auc = results.mean()
                best_epoch = epoch
                torch.save(model, ckpt_path+str(model.name)+"_best.pt")
                print("Saving the best model file at : " +
                      ckpt_path+str(model.name)+"_best.pt")

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {100*running_loss/batch:.4f}.. ")

        if val_dataset != None:
            print(f"Val loss: {100*val_loss/len(val_dataset):.4f}.. "
                  f"Val AUC: {round(results.mean(), 5)}.. ")

            print(f"Best Val loss: {100*best_val_loss:.4f}.. "
                  f"Best Val AUC: {round(best_val_auc, 5)}.."
                  f"Last LR: {scheduler.get_last_lr()[0]:.9f}..")

        running_loss = 0
        batch = 0
        model.train()

    print("-------------------------------------")
    print("..Training is done!")

    if val_dataset != None:
        print("..Best Epoch: ", best_epoch + 1)

    torch.save(model, ckpt_path+"/"+str(model.name)+"_final.pt")
