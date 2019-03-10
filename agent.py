from comet_ml import Experiment
import torch
from datetime import datetime
from tensorboardX import SummaryWriter


class Agent():

    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    def load_checkpoint(self, path):
        self.checkpoint = torch.load(path)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model_state = self.checkpoint['model_state']
        print('Loaded checkpoint at epoch %d' % self.model_state['epochs'])

    def print_accuracy(self):
        print('Training Set:')
        print('Accuracy: %.2f%% | Loss: %.4f'
              % (self.model_state['train_acc'],
                 self.model_state['train_loss']))
        print('Validation Set:')
        print('Accuracy: %.2f%% | Loss: %.4f'
              % (self.model_state['val_acc'],
                 self.model_state['val_loss']))
        print('Test Set:')
        print('Accuracy: %.2f%% | Loss: %.4f'
              % (self.model_state['test_acc'],
                 self.model_state['test_loss']))

    def calculate_accuracy(self, dataset_key, step=0):
        correct = 0
        dataset = getattr(self.dataloader, dataset_key)
        total = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in dataset:
                x = x.cuda()
                y = y.cuda()
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                loss_fn = torch.nn.CrossEntropyLoss().cuda()
                loss = loss_fn(outputs, y)

        acc = float(correct)/total * 100
        self.log_metric(dataset_key+'_loss', loss.item(), step)
        self.log_metric(dataset_key+'_acc', acc, step)
        return acc, loss.item()

    def train(self, loss_fn, num_epochs, optimizer, scheduler=None):
        step = 0
        self.loss_fn = loss_fn
        now = datetime.utcnow().strftime("%d-%m_%H:%M")
        self.experiment_name = '%s_%s_%depochs_%s' \
            % (self.dataloader.name, self.model.name, num_epochs, now)
        self.writer = SummaryWriter('runs/' + self.experiment_name)
        self.model_state = {
            'dataset': self.dataloader.name,
            'model': self.model.name,
            'epochs': 0,
            'train_loss': -1,
            'train_acc': -1,
            'val_loss': -1,
            'val_acc': -1,
            'test_loss': -1,
            'test_acc': -1
        }
        self.experiment = Experiment(
                api_key="bctrYiho1G2hUui2u2BuHKZS3",
                project_name="general",
                workspace="joaoflf"
        )

        for epoch in range(num_epochs):
            print('Epoch %d' % (epoch+1))
            self.model_state['epochs'] = epoch
            running_corrects = 0

            for t, (x, y) in enumerate(self.dataloader.train):
                x = x.cuda()
                y = y.cuda()
                self.model.train()
                optimizer.zero_grad()
                scores = self.model(x)
                loss = loss_fn(scores, y)
                _, preds = torch.max(scores, 1)
                loss.backward()
                optimizer.step()

                running_corrects += torch.sum(preds == y)
                self.log_metric('train_loss', loss.item(), step)

                if (t+1) % 100 == 0:
                    val_acc, val_loss = self.calculate_accuracy('val', step)

                    print('Epoch: %d | Step: %d | Train Loss: %.4f |'
                          ' Val Loss: %.4f | Val acc: %.2f%%' %
                          (epoch+1, step+1, loss.item(),
                              val_loss, val_acc))

                step += 1
                self.experiment.log_parameters(self.model_state)

            train_acc = running_corrects.double() / len(self.dataloader.train)
            self.log_metric('train_acc', train_acc, step)

            if scheduler:
                scheduler.step(loss)
            print('Epoch: %d | Train Loss: %.4f | Train Accuracy: %.2f%%' %
                  (epoch+1, loss.item(), train_acc))
            print('Saving checkpoint for epoch %d' % (epoch+1))
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state': self.model_state
                }, 'checkpoints/'+self.experiment_name+'.pt')

        print('\nTraining Complete, calculating accuracy and loss....')
        self.calculate_accuracy('train', step)
        self.calculate_accuracy('val', step)
        self.calculate_accuracy('test', step)
        self.print_accuracy()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state': self.model_state
            }, 'checkpoints/'+self.experiment_name+'.pt')

    def log_metric(self, metric, value, step):
        self.writer.add_scalars('Metrics', {metric: value}, step)
        self.experiment.log_metric(metric, value, step=step)
        self.model_state[metric] = value
