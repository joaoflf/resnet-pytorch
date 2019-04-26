from comet_ml import Experiment
import torch
from datetime import datetime
from ignite.engine import (Events, create_supervised_trainer,
                           create_supervised_evaluator)
from ignite.metrics import Accuracy, Loss
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Agent():
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log_interval = 10

    def load_checkpoint(self, path):
        self.checkpoint = torch.load(path)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model_state = self.checkpoint['model_state']
        print('Loaded checkpoint at epoch %d' % self.model_state['epochs'])

    def print_accuracy(self):
        print('Training Set:')
        print('Accuracy: %.2f%% | Loss: %.4f' %
              (self.model_state['train_acc'], self.model_state['train_loss']))
        print('Validation Set:')
        print('Accuracy: %.2f%% | Loss: %.4f' % (self.model_state['val_acc'],
                                                 self.model_state['val_loss']))
        print('Test Set:')
        print('Accuracy: %.2f%% | Loss: %.4f' %
              (self.model_state['test_acc'], self.model_state['test_loss']))

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

        acc = float(correct) / total * 100
        self.log_metric(dataset_key + '_loss', loss.item(), step)
        self.log_metric(dataset_key + '_acc', acc, step)
        return acc, loss.item()

    def log_metric(self, metric, value, step):
        self.writer.add_scalars('Metrics', {metric: value}, step)
        self.experiment.log_metric(metric, value, step=step)
        self.model_state[metric] = value

    def train(self, loss_fn, num_epochs, optimizer, scheduler=None):
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
            workspace="joaoflf")

        desc = "ITERATION - loss: {:.2f}"
        pbar = tqdm(
            initial=0,
            leave=False,
            total=len(self.dataloader.train),
            desc=desc.format(0))

        trainer = create_supervised_trainer(self.model, optimizer,
                                            self.loss_fn, self.device)
        evaluator = create_supervised_evaluator(
            self.model,
            metrics={
                'accuracy': Accuracy(),
                'loss': Loss(self.loss_fn)
            },
            device=self.device)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(
                self.dataloader.train) + 1
            if iter % self.log_interval == 0:
                pbar.desc = desc.format(engine.state.output)
                pbar.update(self.log_interval)
            self.log_metric('train_loss', engine.state.output,
                            engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            pbar.refresh()
            evaluator.run(self.dataloader.train)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            tqdm.write(
                "\nTraining Results - Epoch: {}  Accuracy: {:.2f} Loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_loss))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(self.dataloader.val)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            tqdm.write(
                "Val Results - Epoch: {} Accuracy: {:.2f} Loss: {:.2f}".format(
                    engine.state.epoch, avg_accuracy, avg_loss))

            pbar.n = pbar.last_print_n = 0

        @trainer.on(Events.EPOCH_COMPLETED)
        def save_checkpoint(engine):
            self.model_state['epochs'] = engine.state.epoch
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state': self.model_state
            }, 'checkpoints/' + self.experiment_name + '.pt')

        @trainer.on(Events.COMPLETED)
        def calculate_final_metrics(engine):
            print('\nTraining Complete, calculating accuracy and loss....')
            self.calculate_accuracy('train', engine.state.iteration)
            self.calculate_accuracy('val', engine.state.iteration)
            self.calculate_accuracy('test', engine.state.iteration)
            self.print_accuracy()
            pbar.close()
            self.writer.close()
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state': self.model_state
            }, 'checkpoints/' + self.experiment_name + '.pt')

        trainer.run(self.dataloader.train, max_epochs=num_epochs)
