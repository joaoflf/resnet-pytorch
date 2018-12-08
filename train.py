import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime
from tensorboardX import SummaryWriter

from model import ResNetImported
from dataloaders.dogs_dataloader import DogsDataLoader
from dataloaders.cifar10_dataloader import Cifar10DataLoader
from dataloaders.tiny_imagenet_dataloader import TinyImagenetDataLoader
dtype = torch.cuda.FloatTensor


#Tensorboard writer
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
writer = SummaryWriter('runs/'+now)

def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, loss):
    print('Saving checkpoint for epoch %d' % epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss
        }, 'checkpoints/last_checkpoint.pt')
    
def train(model, dataloader, loss_fn, optim, num_epochs=1, print_every=100):
    step = 0
    for epoch in range(num_epochs):
        print('Epoch %d' % epoch)
        model.train()
        for t, (x, y) in enumerate(dataloader.train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(torch.cuda.LongTensor))

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (step % print_every == 0):
                print('t = %d, loss = %.4f' % (step, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss', loss.item(), step)
            step+=1
        #save checkpoint at the end of each epoch
        save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), loss)

#dogs_dataloader = DogsDataLoader()
#cifar10_dataloader = Cifar10DataLoader()
tiny_imagenet_dataloader = TinyImagenetDataLoader()


model = ResNetImported(200).cuda()
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(model.parameters(), lr= 1e-3)

train(model, tiny_imagenet_dataloader, loss_fn, optim, 50, 20)
