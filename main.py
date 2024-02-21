# Entry of the project
import torch
import wandb
from accelerate.utils import set_seed
import numpy as np
import argparse

from train import SimCLR_trainer

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./default_datasets',
                    help='path to dataset')
parser.add_argument('-pretrain-dataset', default='stl10',
                    help='pretrain dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('--backbone', metavar='ARCH', default='simclr_resnet50',
                    choices=['resnet18', 'resnet50', 'simclr_resnet50'],
                    help='model backbone: ' +
                         'default: simclr_resnet50')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-w', '--width-multiplier', default=2, type=int, metavar='N',
                    help='number of width multiplier refer to paper, only available for simclr_resnet50 (default: 2)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total pretrain epochs to run (default: 500)')
parser.add_argument('--fine-tune-epochs', default=100, type=int, metavar='N',
                    help='number of total pretrain epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size in pretrain (default: 256), '
                         'this is NOT the final sample size in loss calculation')
parser.add_argument('--batch-accumulation', default=1, type=int,
                    metavar='N',
                    help='batch accumulation (default: 1), the value cannot lower than 1，'
                         'in loss calculation, the total sample size (2N) equals to '
                         'batch-accumulation * batch-size * 2')
parser.add_argument('-lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate (default: 3e-4)', dest='learning_rate')
parser.add_argument('--wd', '--weight-decay', default=4e-4, type=float,
                    metavar='W', help='weight decay (default: 4e-4)',
                    dest='weight_decay')
parser.add_argument('--finetune-lr', default=0.0001, type=float,
                    metavar='LR', help='fine-tune learning rate (default: 1e-4)')
parser.add_argument('--finetune-batchsize', default=64, type=int,
                    metavar='LR', help='mini-batch size in fine-tune (default: 64)')
parser.add_argument('--seed', default=1919810, type=int,
                    help='seed for initializing training. (default: 1919810)')
parser.add_argument('--features', default=1024, type=int,
                    help='feature dimension of projection head (default: 1024)')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--contrastive-channels', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--wandb-key', default=None, type=str, metavar='N',
                    help='wandb login key. '
                         'wandb is a powerful tool to visualize data & analyze training process '
                         'if you want to disable wandb, just left it to None ')
parser.add_argument('--save-checkpoint', default=100, type=int, metavar='N',
                    help='save checkpoints after \'save-checkpoint\' epochs in pretraining.')


def SimCLR_main():
    args = parser.parse_args()
    print(args)

    # wandb login
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    # Set a seed to reproduce the process
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        set_seed(args.seed)

    # Start training
    agent = SimCLR_trainer(conf=args)
    agent.SimCLR_pretrain()
    agent.SimCLR_classifier_finetune(model_state_dict=f"./CheckPoints/ResNet50_{self.conf.width_multiplier}/backbone_checkpoint_{args.epochs}_{args.width_multiplier}.pt")


if __name__ == '__main__':
    SimCLR_main()
