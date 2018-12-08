import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as transforms

# Sampler to iterate on dataset, given size and start point
class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))
    
    def __len__(self):
        return self.num_samples

NUM_TRAIN = 49000
NUM_VAL = 1000
dtype = torch.cuda.FloatTensor

class Cifar10DataLoader:
    def __init__(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        print('Downloading dataset...')
        train_set = dset.CIFAR10('./dataloaders/datasets/cifar10', train=True, download=True, transform=transform)
        self.train = DataLoader(train_set, batch_size=40, sampler=ChunkSampler(NUM_TRAIN, 0))
        val_set = dset.CIFAR10('./dataloaders/datasets/cifar10', train=True, download=True, transform=transform)
        self.val = DataLoader(val_set, batch_size=40, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
        test_set = dset.CIFAR10('./dataloaders/datasets/cifar10', train=False, download=True, transform=transform)
        self.test = DataLoader(test_set, batch_size=40)
