import torch
from model import ResNetImported
from torch.autograd import Variable
from dataloaders.cifar10_dataloader import Cifar10DataLoader
from dataloaders.dogs_dataloader import DogsDataLoader
from dataloaders.tiny_imagenet_dataloader import TinyImagenetDataLoader

dtype = torch.cuda.FloatTensor
model = ResNetImported(200).type(dtype)
#data_loader = Cifar10DataLoader()
#data_loader = DogsDataLoader()
data_loader = TinyImagenetDataLoader() 

def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded checkpoint at epoch %d' % checkpoint['epoch'])

def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader:
        x_var = Variable(x.type(dtype))
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct)/num_samples
    print('Got %d / %d correct (%.2f%%) \n' % (num_correct, num_samples, 100 * acc))

load_checkpoint('checkpoints/last_checkpoint.pt', model)

print('Train Accuracy:')
check_accuracy(model, data_loader.train)
print('Validation Accuracy:')
check_accuracy(model, data_loader.val)
print('Test Accuracy:')
check_accuracy(model, data_loader.test)
