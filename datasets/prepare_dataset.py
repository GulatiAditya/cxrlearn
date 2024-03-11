import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def _transform(n_px):
    """
    Create a sequence of image transformations.

    Args:
        n_px (int): Size of the output image.

    Returns:
        Compose: Sequence of image transformations.
    """
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def _getImg(datalist, shape):
    """
    Load and preprocess images from a list of file paths.

    Args:
        datalist (list): List of file paths.
        shape (tuple): Shape of the output images.

    Returns:
        Tensor: Tensor containing the preprocessed images.
    """
    data = []
    count = 0
    print("Processing...")
    for x in datalist:
        try:
            imagex = Image.open(x[0])
            preprocess = _transform(shape)
            imagex = preprocess(imagex)
            data.append(imagex[None, :])
            count += 1
        except:
            print("Unable to open this path : "+x[0])
        if count % 1000 == 0:
            print("Processed ", count, " images! ...")
    print("Done! ", count, " images were converted!")
    return torch.cat(data)


class _CXR_Dataset(Dataset):
    """
    Custom dataset for chest X-ray images.

    Args:
        data (DataFrame): DataFrame containing image paths and labels.
        dataset_classes (list): List of target class columns.
        dataset_img_col (list): List of image path columns.
        dataset_name (str): Name of the dataset.
        withoutLabels (bool): Flag indicating if the dataset contains labels.
        prediction_type (str): Type of prediction task ("binary" or "multi").
        reshape_size (tuple): Shape to which images are resized (default is (224, 224)).
    """

    def __init__(self, data, dataset_classes, dataset_img_col, dataset_name, withoutLabels, prediction_type="binary", reshape_size=(224, 224)):
        # Getting x, y
        self.x = _getImg(data[dataset_img_col].values, reshape_size)

        if dataset_classes != None:
            self.y = torch.tensor(data[dataset_classes].values)
            self.labels = dataset_classes

            self.withoutLabels = False
        else:
            self.withoutLabels = True

        self.len = len(self.x)
        self.prediction_type = prediction_type
        self.name = dataset_name
        print("Dataset initialized! Use the info method to know more. \n")

    def __getitem__(self, index):
        if self.withoutLabels:
            sample = self.x[index]
        else:
            sample = self.x[index], self.y[index]
        return sample

    def __len__(self):
        return self.len

    def info(self, makeOneHot=True):
        """
        Display information about the dataset.

        Args:
            makeOneHot (bool): Flag indicating whether to perform one-hot encoding.
        """
        if self.withoutLabels:
            print("-----------" + str(self.name) +
                  " Dataset -------------------------------------")
            print("------------------------------------------------------------------")
            print("Input X: ", self.x.size())
            print("Prepared dataset without labels")
            return

        print("CXR Dataset classes are as follows: ")

        is_binary = []
        for idx, c in enumerate(self.labels):

            labels_in = sorted((torch.unique(self.y[:, idx])).tolist())
            print("Label header - ", c)
            print("Set of labels - ", labels_in)
            if (len(labels_in) <= 2):
                print("Label "+str(c)+" is Binary Class!")
                is_binary.append(1)
            else:
                print("Label "+str(c)+" is Multi Class!")
                is_binary.append(0)

        assert (sum(is_binary) == 0 and len(is_binary) == 1) or sum(is_binary) == len(
            self.labels), "Supported set of target-class configurations are : One/All binary target columns or Exactly one multiclass column "

        if sum(is_binary) == len(self.labels):
            print("Dataset " + self.name + " is a " +
                  str(len(self.labels)) + "-column binary-class dataset")

        elif (sum(is_binary) == 0 and len(is_binary) == 1):
            print("Dataset " + self.name +
                  " is a single column multi-class dataset")

        self.is_binary = is_binary

        if makeOneHot and self.prediction_type == "multi":
            self._makeonehot()

        print("------------------------------------------------------------------")
        print("-----------" + str(self.name) +
              " Dataset -------------------------------------")
        print("------------------------------------------------------------------")
        print("Input X: ", self.x.size())
        print("Output Y: ", self.y.size())
        print("------------------------------------------------------------------")
        print(
            "---Note: In case you see set of labels as [0,1] it can be due to one-hot labeling")

    def _makeonehot(self):
        """
        Perform one-hot encoding for multi-class datasets.
        """
        if self.is_binary[0] == 0:
            labels_in = sorted((torch.unique(self.y[:, 0])).tolist())
            nClass = len(set(labels_in))
            idx_in = list(range(nClass))
            list_labels = list(set(labels_in))
            print(list_labels)

            y_onehot = []
            for i in list(self.y):
                onehot = [0]*nClass
                onehot[list_labels.index(i)] = 1
                y_onehot.append(onehot)
            self.y = torch.tensor(y_onehot)

    def save(self, PATH):
        """
        Save the dataset to a .pt file.

        Args:
            PATH (str): Path where the .pt file will be saved.
        """
        print("Saving dataset to ", PATH, " ...")
        torch.save(self, PATH+self.name+".pt")
        print("Saved pt file...")


def _cxr_to_pt(csv_file, path_col, class_col, dataset_name, out_pt_address, reshape_size=(224, 224), pt_withoutLabels=False, prediction_type="binary", skip_loading=False, fewshot_class=None, fewshot_perclass=0, save_data=True):
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
    if pt_withoutLabels:
        dataset_name = dataset_name + "-NoLabels"

    if fewshot_perclass > 0:
        dataset_name = "fewshot"+str(fewshot_perclass)+"-"+dataset_name

    dataset_name = dataset_name + "-" + \
        str(reshape_size[0]) + "-" + \
        str(reshape_size[1]) + "-" + prediction_type

    if os.path.exists(out_pt_address+dataset_name+".pt") == True and skip_loading == False:
        print(".pt file already exists at - " +
              str(out_pt_address+dataset_name)+".pt"+".......")
        print("Loading pt file directly into object.....")
        data = _load_CXR(out_pt_address+dataset_name+".pt", dataset_name)
        data.info(makeOneHot=False)
        return data

    assert os.path.exists(csv_file), str(csv_file) + " does not exist."

    data_ids = pd.read_csv(csv_file)

    if fewshot_class == None:
        fewshot_class = class_col

    if fewshot_perclass > 0:
        data_ids = data_ids.groupby(fewshot_class).apply(
            lambda x: x.sample(fewshot_perclass, replace=True))
        data_ids = data_ids.drop_duplicates(subset=path_col)

    assert len(path_col) == 1, "Only one column for path is supported."
    assert set(path_col).issubset(
        data_ids.columns), "path column doesnt exist."

    classes = None

    if not pt_withoutLabels:
        assert set(class_col).issubset(
            data_ids.columns), "target class columns dont exist"
        assert len(
            class_col) > 0, "Atleast one target column is needed (incase of a training set), otherwise mark withoutLables=True"
        classes = class_col

    img_col = path_col

    data = _CXR_Dataset(data_ids, classes, img_col, dataset_name, pt_withoutLabels,
                        prediction_type=prediction_type, reshape_size=reshape_size)

    if prediction_type == "multi":
        data.info(makeOneHot=True)
    else:
        data.info(makeOneHot=False)

    if not os.path.exists(out_pt_address):
        os.path.makedirs(out_pt_address)

    if save_data:
        data.save(out_pt_address)

    return data


def _load_CXR(PATH, dataset_name):
    """
    Load a .pt file containing a CXR dataset.

    Args:
        PATH (str): Path to the .pt file.
        dataset_name (str): Name of the dataset.

    Returns:
        _CXR_Dataset: Loaded dataset.
    """
    print("Loading .pt files......")
    data = torch.load(PATH)
    print("Loaded pt files for "+str(dataset_name)+" dataset")
    # data.name = data.name + "-" + prediction_type
    return data
