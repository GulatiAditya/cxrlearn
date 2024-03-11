import cxrlearn
import torch
from args import BaseArgParser

# Function to run the training process


def run(args):
    # Extract arguments
    data_args = args.data_args
    model_args = args.model_args
    optim_args = args.optim_args

    # Print training details
    print("Training" + model_args.model)
    print("Learning rate: " + str(optim_args.lr))

    # Set device based on availability
    device = torch.device(
        f"cuda:{args.gpu_ids}" if torch.cuda.is_available() else "cpu")

    # Convert training, validation, and test datasets to PyTorch format if provided

    train = cxrlearn.cxr_to_pt(data_args.train_csv, [data_args.path_column], data_args.class_columns,
                               data_args.name+"-train", data_args.datapt_dir) if data_args.train_csv else None
    val = cxrlearn.cxr_to_pt(data_args.val_csv, [data_args.path_column], data_args.class_columns,
                             data_args.name+"-val", data_args.datapt_dir) if data_args.val_csv else None
    test = cxrlearn.cxr_to_pt(data_args.val_csv, [data_args.path_column], data_args.class_columns,
                              data_args.name+"-test", data_args.datapt_dir) if data_args.test_csv else None

    # Initialize the model based on model argument
    if model_args.model == "medaug":
        model = cxrlearn.medaug(model=model_args.arch, pretrained_on=model_args.pretrained_on,
                                device=device, freeze_backbone=optim_args.freeze_backbone, num_out=data_args.num_classes)
    elif model_args.model == "cxr-repair":
        model = cxrlearn.chexzero(device=device, freeze_backbone=optim_args.freeze_backbone,
                                  linear_layer_dim=model_args.cxr_repair_layer_dim, num_out=data_args.num_classes)
    elif model_args.model == "refers":
        model = cxrlearn.refers(
            device=device, freeze_backbone=optim_args.freeze_backbone, num_out=data_args.num_classes)
    elif model_args.model == "convirt":
        model = cxrlearn.convirt(
            device=device, freeze_backbone=optim_args.freeze_backbone, num_out=data_args.num_classes)
    elif model_args.model == "gloria":
        model = cxrlearn.gloria(model=model_args.arch, device=device, freeze_backbone=optim_args.freeze_backbone,
                                num_ftrs=model_args.gloria_layer_dim, num_out=data_args.num_classes)
    elif model_args.model == "s2mts2":
        model = cxrlearn.s2mts2(
            device=device, freeze_backbone=optim_args.freeze_backbone, num_out=data_args.num_classes)
    elif model_args.model == "mococxr":
        model = cxrlearn.mococxr(model=model_args.arch, device=device,
                                 freeze_backbone=optim_args.freeze_backbone, num_out=data_args.num_classes)
    elif model_args.model == "resnet":
        model = cxrlearn.resnet(device=device, num_out=data_args.num_classes)
    else:
        return

    # Set model name based on training dataset and model type
    model.name = train.name + "-" + model.name

    # Train the model
    if train:
        if optim_args.freeze_backbone and optim_args.use_logreg:
            cxrlearn.finetune_logreg(train_dataset=train, test_dataset=test,
                                     model=model, device=device, max_iter=optim_args.max_iter)

        else:
            cxrlearn.finetune(train_dataset=train, val_dataset=val, model=model, device=device, batch_size=data_args.batch_size,
                              epochs=optim_args.num_epochs, lr=optim_args.lr, momentum=optim_args.momentum, ckpt_path=model_args.chk_dir)

    # Evaluate the model if test dataset is provided and not using logistic regression
    if test and not optim_args.use_logreg:
        print("------------Result for Best Epoch-------------------")
        cxrlearn.evaluate_fromckpts(test_dataset=test, ckpt_path=model_args.chk_dir+str(
            model.name)+"_best.pt", device=device, epochs=optim_args.num_epochs, lr=optim_args.lr)


# Main function
if __name__ == "__main__":
    # Parse command line arguments
    parser = BaseArgParser()
    run(parser.parse_args())
