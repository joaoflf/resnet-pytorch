import torch
import argparse
from agent import Agent
from models.resnet import ResNet34, ResNet50
from models.resnet_imported import ResNetImported
from dataloaders.dogs_dataloader import DogsDataLoader
from dataloaders.cifar10_dataloader import Cifar10DataLoader
from dataloaders.tiny_imagenet_dataloader import TinyImagenetDataLoader


parser = argparse.ArgumentParser(description='Accuracy Checker')
parser.add_argument('--dataset', metavar='DATASET', default='tiny_imagenet',
                    choices=['tiny_imagenet', 'dogs', 'cifar10'],
                    help='name of dataset  (default: tiny_imagenet)')
parser.add_argument('--model', metavar='MODEL', default='resnet34',
                    choices=['resnet34', 'resnet50', 'resnet50-imported',
                             'resnet50-preact'],
                    help='name of model arch  (default: resnet34)')
parser.add_argument('--checkpoint', metavar="CHECKPOINT",
                    help='path to your model checkpoint')


def main():
    args = parser.parse_args()
    model = None
    dtype = torch.cuda.FloatTensor

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
        model = ResNet34(num_classes).type(dtype)
    elif args.model == 'resnet50':
        model = ResNet50(num_classes).type(dtype)
    elif args.model == 'resnet50-preact':
        model = ResNet50(num_classes, pre_activation=True).type(dtype)
    elif args.model == 'resnet50-imported':
        model = ResNetImported(num_classes).type(dtype)

    agent = Agent(model, dataloader)
    agent.load_checkpoint(args.checkpoint)
    # agent.check_accuracy()
    agent.calculate_accuracy('test')


if __name__ == '__main__':
    main()
