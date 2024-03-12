# cxrlearn

cxrlearn is based on the work "Self-Supervised Pretraining Enables High-Performance Chest X-Ray Interpretation Across Clinical Distributions" [(Paper Link)](https://www.medrxiv.org/content/10.1101/2022.11.19.22282519v1).

cxrleanr is a library for self-supervised learning, finetuning, and probing on chest X-ray images. It supports various state-of-the-art self-supervised models, enabling high-performance chest X-ray interpretation across clinical distributions. With cxrlearn, you can effortlessly train and evaluate models for various downstream tasks in chest X-ray analysis.

## Prerequisites

### 1. Pretrained Weights
Download the pretrained.zip from the provided Google Drive link using the following commands:

```
pip install gdown
gdown 10PU5-m27Ph-VxpS-aHOHYN_njD38Xbme
cd pretrained
unzip pretrained.zip
```
<!--- gdown https://drive.google.com/drive/folders/0ByJvtNcqeRr4VFd6TTYyYW8wYkE?resourcekey=0-yi-xFTNM7msW4fGYJ0uhOA&usp=sharing --->

Alternatively, you can manually download the pretrained.zip into the 'pretrained' folder from Google Drive. [Google Drive Link](https://drive.google.com/drive/folders/0ByJvtNcqeRr4VFd6TTYyYW8wYkE?resourcekey=0-yi-xFTNM7msW4fGYJ0uhOA&usp=sharing)

### 2. Environment setup
Create and setup the Anaconda virtual environment using the provided environment.yml file:
```
conda env create -f environment.yml
conda activate cxr
```

### 3. Usage 

cxrlearn can be utilized both as a library and through command line arguments for self-supervised learning finetuning. Below are examples demonstrating how to use CXRLearn for linear finetuning and full finetuning:

Format
```
python3 run_cxr.py --gpu_ids [GPU_IDS] --train_csv [PATH_TO_CSV] --val_csv [PATH_TO_CSV] --test_csv [PATH_TO_CSV] --path_column [PATH_COLUMN_NAME] --class_columns [PREDICTION_COLUMN_NAMES] --datapt_dir [PATH_TO_DIR] --num_classes [NUMBER_OF_CLASSES] --name [EXPERIMENT_NAME] --freeze_backbone [FREEZE_FLAG] --use_logreg [LOGISTIC_REGRESSION_FLAG] --max_iter [MAX_ITERATIONS_LOGISTIC_REGRESSION] --chk_dir [PATH_TO_DIR] --model [MODEL_NAME] --arch [ARCHITECTURE] --cxr_repair_layer_dim [LAYER_DIMENSION_IN_CXR_REPAIR] --gloria_layer_dim [GLORIA_LAYER_DIMENSION] --pretrained_on [PRETRAINING_DATASET] --batch_size [BATCH_SIZE] --lr [LEARNING_RATE] --num_epochs [NUMBER_OF_EPOCHS] --momentum [MOMENTUM]
```

Example for linear fine-tuning

```
python3 run_cxr.py --gpu_ids 0 \
				   --train_csv "/path/to/train.csv" \
				   --val_csv "/path/to/valid.csv" \
				   --test_csv "/path/to/test.csv" \
				   --path_column "Path" \
				   --class_columns "Atelectasis" "Cardiomegaly" "Effusion" "Infiltration" "Mass" "Nodule" "Pneumonia" "Pneumothorax" "Consolidation" "Edema" "Emphysema" "Fibrosis" "Pleural_Thickening" "Hernia" \
				   --datapt_dir "./pt-dataset/" \
				   --num_classes 14 \
				   --name cxr14 \
				   --freeze_backbone True \
				   --use_logreg True \
				   --max_iter 500 \
				   --chk_dir "./pt-finetune" \
				   --model gloria \
				   --arch "resnet50" \
				   --gloria_layer_dim 2048
```
Example for full fine-tuning
```
python3 run_cxr.py --gpu_ids 0 \
				   --train_csv "/path/to/train.csv" \
				   --val_csv "/path/to/valid.csv" \
				   --test_csv "/path/to/test.csv" \
				   --path_column "Path" \
				   --class_columns "Atelectasis" "Cardiomegaly" "Effusion" "Infiltration" "Mass" "Nodule" "Pneumonia" "Pneumothorax" "Consolidation" "Edema" "Emphysema" "Fibrosis" "Pleural_Thickening" "Hernia" \
				   --datapt_dir "./pt-dataset/" \
				   --num_classes 14 \
				   --name cxr14 \
				   --freeze_backbone False \
				   --use_logreg False \
				   --chk_dir "./pt-finetune" \
				   --model gloria \
				   --arch "resnet50" \
				   --gloria_layer_dim 2048 \
				   --pretrained_on "mimic-cxr" \
				   --batch_size 32 \
				   --lr 1e-3 \
				   --num_epochs 100 \
				   --momentum 0.9


```



### 4. Supported Models
The library supports finetuning, linear probing, and logistic regression based probing for 7 pretrained Self-Supervised Models pretrained on chest x-ray images. The supported SSL based models are:

1. MedAug - MedAug: Contrastive learning leveraging patient metadata improves representations for chest X-ray interpretation https://arxiv.org/abs/2102.10663
2. CheXzero - https://github.com/stanfordmlgroup/aihc-win21-clip
3. REFERS - Generalized radiograph representation learning via cross-supervision between images and free-text radiology reports https://www.nature.com/articles/s42256-021-00425-9.pdf
4. ConVIRT - CONTRASTIVE LEARNING OF MEDICAL VISUAL REPRESENTATIONS FROM PAIRED IMAGES AND TEXT https://arxiv.org/pdf/2010.00747.pdf
5. S2MTS2 - Self-supervised Mean Teacher for Semi-supervised Chest X-ray Classification https://arxiv.org/pdf/2103.03629.pdf
6. GLoRIA - GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.pdf
7. MocoCXR - MoCo-CXR: MoCo Pretraining Improves Representation and Transferability of Chest X-ray Models https://arxiv.org/pdf/2010.05352.pdf

### 5. Results

The results obtained from finetuning and linear probing are benchmarked across various ChestXray downstream tasks. Below are the AUROC scores:

#### Finetuning Results
![Finetuning Results](https://github.com/GulatiAditya/cxrlearn/blob/main/imgs/finetuning.png)

#### Linear Probing Results
![Linear Probing Results](https://github.com/GulatiAditya/cxrlearn/blob/main/imgs/linear_probing.png)

### 6. References

cxrlearn is based on the work "Self-Supervised Pretraining Enables High-Performance Chest X-Ray Interpretation Across Clinical Distributions" [(Paper Link)](https://www.medrxiv.org/content/10.1101/2022.11.19.22282519v1).





