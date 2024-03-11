"""Define base class for processing command-line arguments."""
import argparse
import copy
# import json
# from pathlib import Path

# from .. import util
# from constants import *


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='cxrlearn')

        self.parser.add_argument('--gpu_ids',
                                 type=str, default='0',
                                 help=('Comma-separated list of GPU IDs. ' +
                                       'Use -1 for CPU.'))

        self.parser.add_argument('--train_csv', type=str, dest="data_args.train_csv", required=True, 
                                 help="Training CSV path - csvs should have images path, and labels")

        self.parser.add_argument('--val_csv', type=str, dest="data_args.val_csv", required=False, 
                                 help="Validation CSV path - csvs should have images path, and labels")

        self.parser.add_argument('--test_csv', type=str, dest="data_args.test_csv", required=False, 
                                 help="Test CSV path - csvs should have images path and \
                                        optionally labels for evaluation")
        
        self.parser.add_argument('--path_column',
                                 dest='data_args.path_column',type=str, required=True,
                                 help="Name of column containing images path")

        self.parser.add_argument('--class_columns', nargs="*", type=str,
                                 dest='data_args.class_columns', required=True, 
                                 help="Name of columns with class information")

        self.parser.add_argument('--datapt_dir',dest='data_args.datapt_dir',
                                 type=str, default='./pt-dataset/',
                                 help='Directory to save pt versions of datasets (for fast loading)')
        
        self.parser.add_argument('--num_classes',dest='data_args.num_classes',
                                 type=int, required=True, help='Number of classes (depends on the setting of problem binary vs multiclass vs multilabel')

        self.parser.add_argument('--name',dest='data_args.name',type=str, default='data_default',
                                 help='Give name to the experiment')
        
        self.parser.add_argument('--freeze_backbone', dest='optim_args.freeze_backbone',action='store_true',
                                help='Linear vs Finetuning (True/False) - default is True')

        self.parser.add_argument('--no-freeze_backbone', dest='freeze_backbone',action='store_false')
        self.parser.set_defaults(freeze_bacbone=True)

        self.parser.add_argument('--use_logreg',
                                dest="optim_args.use_logreg",
                                action="store_true",
                                help='Use Logreg in Linear Tuning')

        self.parser.add_argument('--no-use_logreg', dest='use_logreg',action='store_false')
        self.parser.set_defaults(use_logreg=True)

        self.parser.add_argument('--max_iter',
                                dest='optim_args.max_iter',
                                type=int,
                                default=500,
                                help='Maximum number of iterations for logistic regression model')

        self.parser.add_argument('--chk_dir',
                                 dest='model_args.chk_dir',
                                 type=str, default="./pt-finetune/",
                                 help='Directory to save models checkpoints.')

        self.parser.add_argument('--model',dest='model_args.model',
                                 type=str, default='convirt', choices= ["medaug", "cxr-repair", "refers", "convirt", "gloria", "s2mts2", "mococxr", "resnet"],
                                 help='Select a model from -- medaug, cxr-repair, refers, convirt, gloria, s2mts2, mococxr, resnet')

        self.parser.add_argument('--arch', dest='model_args.arch', type=str, default="resnet50",
                                 choices=["resnet50", "resnet18"],
                                help="Backbone architecture choose -- resnet50 or resnet18 (valid only for medaug, mococxr and gloria)")

        self.parser.add_argument('--cxr_repair_layer_dim', dest='model_args.cxr_repair_layer_dim', type=int, default=512,
                                    help="Choose hidden layer dimension for cxr repair (default 512).")
        
        self.parser.add_argument('--gloria_layer_dim', dest='model_args.gloria_layer_dim', type=int, default=2048,
                                    help="Choose hidden layer dimension for gloria (default 2048).")
        
        self.parser.add_argument('--pretrained_on', dest='model_args.pretrained_on', type=str, default="mimic-cxr",choices=["mimic-cxr","chexpert"],
                                    help="Pretraining dataset -- mimic-cxr or chexpert (valid only for medaug)")


        self.parser.add_argument('--batch_size',
                                 dest='data_args.batch_size',
                                 type=int, default=32,
                                 help='Batch size for training / evaluation.')

        self.parser.add_argument('--lr',
                                 dest='optim_args.lr',
                                 type=float, default=1e-3,
                                 help='Initial learning rate.')
        
        self.parser.add_argument('--num_epochs',
                                 dest='optim_args.num_epochs',
                                 type=int, default=100,
                                 help=('Number of epochs to train. If 0, ' +
                                       'train forever.'))

        self.parser.add_argument('--momentum',
                                 dest='optim_args.momentum',
                                 type=float, default=0.9,
                                 help='Momentum')

    def namespace_to_dict(self, args):
        """Turns a nested Namespace object to a nested dictionary"""
        args_dict = vars(copy.deepcopy(args))

        for arg in args_dict:
            obj = args_dict[arg]
            if isinstance(obj, argparse.Namespace):
                args_dict[arg] = self.namespace_to_dict(obj)

        return args_dict

    def fix_nested_namespaces(self, args):
        """Makes sure that nested namespaces work
            Args:
                args: argsparse.Namespace object containing all the arguments
            e.g args.data_args.batch_size

            Obs: Only one level of nesting is supported.
        """
        group_name_keys = []

        for key in args.__dict__:
            if '.' in key:
                group, name = key.split('.')
                group_name_keys.append((group, name, key))

        for group, name, key in group_name_keys:
            if group not in args:
                args.__dict__[group] = argparse.Namespace()

            args.__dict__[group].__dict__[name] = args.__dict__[key]
            del args.__dict__[key]

    def parse_args(self):
        """Parse command-line arguments and set up directories and other run
        args for training and testing."""
        args = self.parser.parse_args()

        self.fix_nested_namespaces(args)
        return args
        
