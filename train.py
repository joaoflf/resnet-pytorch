import argparse
from agent import Agent
import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet import ResNet34, ResNet50
from models.resnet_imported import ResNetImported
from models.resnext import ResNext29
from models.resnext29 import resnext29_8x64d
from dataloaders.dogs_dataloader import DogsDataLoader
from dataloaders.cifar10_dataloader import Cifar10DataLoader
from dataloaders.tiny_imagenet_dataloader import TinyImagenetDataLoader

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument(
    '--epochs',
    default=50,
    type=int,
    metavar='N',
    help='number of total epochs to run'
)
parser.add_argument(
    '--dataset',
    metavar='DATASET',
    default='tiny_imagenet',
    choices=['tiny_imagenet', 'dogs', 'cifar10'],
    help='name of dataset  (default: tiny_imagenet)'
)
parser.add_argument(
    '--model',
    metavar='MODEL',
    default='resnet34',
    choices=[
        'resnet34',
        'resnet50',
        'resnet50-imported',
        'resnet50-preact',
        'resnext29'
    ],
    help='name of model arch  (default: resnet34)'
)
parser.add_argument(
    '--optimizer',
    metavar='OPTIM',
    default='momentum',
    choices=['momentum', 'adam'],
    help='name of model arch  (default: resnet34)'
)
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate',
    dest='lr'
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    metavar='M',
    help='momentum'
)
parser.add_argument(
    '--wd',
    '--weight-decay',
    default=1e-4,
    type=float,
    metavar='W', help='weight decay (default: 1e-4)',
    dest='weight_decay'
)
parser.add_argument(
    '--log-and-save',
    default=False,
    type=bool,
    metavar='L',
    help='log data and save checkpoints'
)


def main():
    args = parser.parse_args()
    model = None
    scheduler = None

    if args.dataset == 'tiny_imagenet':
        dataloader = TinyImagenetDataLoader()
        num_classes = 200
    elif args.dataset == 'dogs':
        dataloader = DogsDataLoader()
        num_classes = 121
    else:
        dataloader = Cifar10DataLoader()
        num_classes = 10

    if args.model == 'resnet34':
        model = ResNet34(num_classes).cuda()
    elif args.model == 'resnet50':
        model = ResNet50(num_classes).cuda()
    elif args.model == 'resnet50-preact':
        model = ResNet50(num_classes, pre_activation=True).cuda()
    elif args.model == 'resnet50-imported':
        model = ResNetImported(num_classes).cuda()
    elif args.model == 'resnext29':
        # model = ResNext29(num_classes).cuda()
        model = resnext29_8x64d(num_classes).cuda()

    if args.optimizer == 'momentum':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.1
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = nn.CrossEntropyLoss().cuda()
    agent = Agent(model, dataloader)
    agent.train(loss_fn, args.epochs, optimizer, scheduler, args.log_and_save)


if __name__ == '__main__':
    main()
