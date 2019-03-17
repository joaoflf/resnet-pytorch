import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from scipy.io import loadmat
from PIL import Image


class DogsDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        if train:
            file = loadmat(root_dir + '/train_list.mat')
        else:
            file = loadmat(root_dir + '/test_list.mat')

        images = file['file_list'].flatten()
        self.images = [
            root_dir + '/Images/' + val for sublist in images
            for val in sublist
        ]
        self.labels = file['labels'].astype(int).flatten()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_name = self.images[index]
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return (image, self.labels[index])


# Sampler to iterate on dataset, given size and start point
class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


NUM_TRAIN = 11000
NUM_VAL = 1000
dtype = torch.cuda.FloatTensor


class DogsDataLoader():
    def __init__(self):
        transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor()])

        self.name = 'dogs'
        dataset = DogsDataset(
            './dataloaders/datasets/dogs', transform=transform)
        self.train = DataLoader(
            dataset, batch_size=40, sampler=ChunkSampler(NUM_TRAIN, 0))
        self.val = DataLoader(
            dataset, batch_size=40, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
        dataset_test = DogsDataset(
            './dataloaders/datasets/dogs', transform=transform, train=False)
        self.test = DataLoader(dataset_test, batch_size=40)
