import torch
import torch.nn as nn
import torchvision.models as models

#imported resnet50
class ResNetImported(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        model = models.resnet50(pretrained=pretrained)
        self.name = 'ResNet50Imported'
        #if pretrained, freeze layers
        if pretrained:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            #set last layer to output the number of classes
            model.fc = nn.Linear(model.fc.in_features, 1000)
            self.model = model
            self.extra_layers = nn.Sequential(
                    nn.BatchNorm1d(1000),
                    nn.ReLU(),
                    nn.Linear(1000, 1000),
                    nn.BatchNorm1d(1000),
                    nn.ReLU(),
                    nn.Linear(1000, 500),
                    nn.BatchNorm1d(500),
                    nn.ReLU(),
                    nn.Linear(500, num_classes)
                )
        else: 
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        self.model = model

    def forward(self, x):
        if self.pretrained:
            return self.extra_layers(self.model(x))
        else:
            return self.model(x)


