import torch
from models.resnet import ResNet50
from torch.autograd import Variable
from dataloaders.cifar10_dataloader import Cifar10DataLoader
from dataloaders.dogs_dataloader import DogsDataLoader
from dataloaders.tiny_imagenet_dataloader import TinyImagenetDataLoader

dtype = torch.cuda.FloatTensor
model = ResNet50(10).type(dtype)
data_loader = Cifar10DataLoader()
#data_loader = DogsDataLoader()
#data_loader = TinyImagenetDataLoader() 
def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded checkpoint at epoch %d' % checkpoint['model_state']['current_epoch'])

def check_accuracy(model, loader):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            loss_fn = torch.nn.CrossEntropyLoss().type(dtype)
            loss = loss_fn(outputs, y)

    acc = float(correct)/total * 100
    print('Got %d / %d correct (%.2f%%) | Loss: %.4f \n' %
            (correct, total, acc, loss.item()))

load_checkpoint('checkpoints/cifar10_ResNet50_50epochs_07-01_11:54.pt', model)

print('Train Accuracy:')
check_accuracy(model, data_loader.train)
print('Validation Accuracy:')
check_accuracy(model, data_loader.val)
print('Test Accuracy:')
check_accuracy(model, data_loader.test)
