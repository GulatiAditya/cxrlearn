# cxrlearn
# Prerequisites
### 1. Pretrained weights
Download the pretrained.zip from the google drive location, using following set of commands.

```
pip install gdown
gdown 10PU5-m27Ph-VxpS-aHOHYN_njD38Xbme
```
<!--- gdown https://drive.google.com/drive/folders/0ByJvtNcqeRr4VFd6TTYyYW8wYkE?resourcekey=0-yi-xFTNM7msW4fGYJ0uhOA&usp=sharing --->

Go into the newly created pretrained directory and unzip the pretrained.zip
```
cd pretrained
unzip pretrained.zip
```

Or download the pretrained.zip into the pretrained folder manually from gdrive link (https://drive.google.com/drive/folders/0ByJvtNcqeRr4VFd6TTYyYW8wYkE?resourcekey=0-yi-xFTNM7msW4fGYJ0uhOA&usp=sharing)

### 2. Environment setup
Create and setup the anaconda virtual environment using the following set of commands.
```
conda env create -f environment.yml
conda activate cxr
```

# cxr-learn
The library supports finetuning, linear probing, and logistic regression based probing for 7 pretrained Self-Supervised Models pretrained on chest x-ray images. The support SSL based models are:

1. MedAug - MedAug: Contrastive learning leveraging patient metadata improves representations for chest X-ray interpretation https://arxiv.org/abs/2102.10663
2. CheXzero - https://github.com/stanfordmlgroup/aihc-win21-clip
3. REFERS - Generalized radiograph representation learning via cross-supervision between images and free-text radiology reports https://www.nature.com/articles/s42256-021-00425-9.pdf
4. ConVIRT - CONTRASTIVE LEARNING OF MEDICAL VISUAL REPRESENTATIONS FROM PAIRED IMAGES AND TEXT https://arxiv.org/pdf/2010.00747.pdf
5. S2MTS2 - Self-supervised Mean Teacher for Semi-supervised Chest X-ray Classification https://arxiv.org/pdf/2103.03629.pdf
6. GLoRIA - GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.pdf
7. MocoCXR - MoCo-CXR: MoCo Pretraining Improves Representation and Transferability of Chest X-ray Models https://arxiv.org/pdf/2010.05352.pdf

# Examples

Format
```
python3 run_cxr.py --gpu_ids [GPU_IDS] --train_csv [PATH_TO_CSV] --val_csv [PATH_TO_CSV] --test_csv [PATH_TO_CSV] --path_column [PATH_COLUMN_NAME] --class_columns [PREDICTION_COLUMN_NAMES] --datapt_dir [PATH_TO_DIR] --num_classes [NUMBER_OF_CLASSES] --name [EXPERIMENT_NAME] --freeze_backbone [FREEZE_FLAG] --use_logreg [LOGISTIC_REGRESSION_FLAG] --max_iter [MAX_ITERATIONS_LOGISTIC_REGRESSION] --chk_dir [PATH_TO_DIR] --model [MODEL_NAME] --arch [ARCHITECTURE] --cxr_repair_layer_dim [LAYER_DIMENSION_IN_CXR_REPAIR] --gloria_layer_dim [GLORIA_LAYER_DIMENSION] --pretrained_on [PRETRAINING_DATASET] --batch_size [BATCH_SIZE] --lr [LEARNING_RATE] --num_epochs [NUMBER_OF_EPOCHS] --momentum [MOMENTUM]
```

Example for linear fine-tuning

```
python3 run_cxr.py --gpu_ids 0 \
				   --train_csv "/deep2/group/cxr-transfer/datasets/cxr14/train.csv" \
				   --val_csv "/deep2/group/cxr-transfer/datasets/cxr14/valid.csv" \
				   --test_csv "/deep2/group/cxr-transfer/datasets/cxr14/test.csv" \
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
				   --gloria_layer_dim 2048 \
```
Example for full fine-tuning
```
python3 run_cxr.py --gpu_ids 0 \
				   --train_csv "/deep2/group/cxr-transfer/datasets/cxr14/train.csv" \
				   --val_csv "/deep2/group/cxr-transfer/datasets/cxr14/valid.csv" \
				   --test_csv "/deep2/group/cxr-transfer/datasets/cxr14/test.csv" \
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

