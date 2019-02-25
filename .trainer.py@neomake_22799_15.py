import torch
from datetime import datetime
from tensorboardX import SummaryWriter

class Trainer():

    def __init__(self, model, dataloader, loss_fn, num_epochs,
            optimizer, scheduler=None):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn.cuda()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        now = datetime.utcnow().strftime("%d-%m_%H:%M")
        self.name = '%s_%s_%depochs_%s' % (self.dataloader.name, self.model.name,
                                                self.num_epochs, now)
        self.writer = SummaryWriter('runs/' + self.name)
        self.model_state = {
            'dataset': self.dataloader.name,
            'model':self.model.name,
            'current_epoch': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1
        }
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


    def check_validation_set(self):
        val_corrects = 0
        self.model.eval()

        for t, (x, y) in enumerate(self.dataloader.val):
            x = x.cuda()
            y = y.cuda()
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                scores = self.model(x)
                val_loss = self.loss_fn(scores, y)
                _, preds = torch.max(scores, 1)
                val_corrects += torch.sum(preds == y)
        return val_loss, val_corrects

    def check_test_set(self):
        print('Checking test set accuracy...')
        test_corrects = 0
        total = len(self.dataloader.test)
        self.model.eval()

        for t, (x, y) in enumerate(self.dataloader.test):
            x = x.cuda()
            y = y.cuda()
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                scores = self.model(x)
                test_loss = self.loss_fn(scores, y)
                _, preds = torch.max(scores, 1)
                test_corrects += torch.sum(preds == y)
        test_acc = test_corrects.double() / total
        print('Test Set : got %d / %d correct (%.2f%%) | Test Loss: %.4f \n' %
                (test_corrects, total, test_acc * 100, test_loss.item()))

    def train(self):
        step = 0

        for epoch in range(self.num_epochs):
            print('Epoch %d' % (epoch+1))

            self.model_state['current_epoch'] = epoch
            running_corrects=0
            for t, (x, y) in enumerate(self.dataloader.train):
                x = x.cuda()
                y = y.cuda()
                self.model.train()
                self.optimizer.zero_grad()

                scores = self.model(x)
                loss = self.loss_fn(scores, y)
                _, preds = torch.max(scores, 1)
                loss.backward()

                self.optimizer.step()

                running_corrects += torch.sum(preds == y)
                self.writer.add_scalars('Metrics', {'loss': loss.item()}, step)

                if (t+1) % 100 == 0:
                    val_loss, val_corrects = self.check_validation_set()
                    val_acc = val_corrects.double() / len(self.dataloader.val)

                    self.writer.add_scalars('Metrics',
                            {'val_loss':val_loss.item(), 'val_accuracy':val_acc}, step)
                    self.model_state['train_loss'].append(loss.item())
                    self.model_state['val_loss'].append(val_loss.item())
                    self.model_state['val_loss'].append(val_acc.item())

                    print('Epoch: %d | Step: %d | Train Loss: %.4f |'
                            ' Val Loss: %.4f | Val acc: %.2f%%' %
                            (epoch+1, step+1, loss.item(), val_loss.item(), val_acc))
                step+=1

            train_acc = running_corrects.double() / len(self.dataloader.train)
            self.writer.add_scalars('Metrics', {'train_accuracy':train_acc}, step)
            self.model_state['train_acc'].append(train_acc)

            if self.scheduler:
                self.scheduler.step(loss)
            print('Epoch: %d | Train Loss: %.4f | Train Accuracy: %.2f%%' %
                  (epoch+1, loss.item(), train_acc))
            print('Saving checkpoint for epoch %d' % (epoch+1))
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state': self.model_state
                }, 'checkpoints/'+self.name+'.pt')
        self.check_test_set()

