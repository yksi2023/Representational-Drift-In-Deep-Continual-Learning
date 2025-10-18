import torch
from torch import nn

class FashionMNISTModel(nn.Module):
    '''A simple feedforward neural network for FashionMNIST classification.'''
    def __init__(self, input_size=784, hidden_size=[256], output_size=10):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    
class ResNet18(nn.Module):
    '''ResNet18 model adapted for FashionMNIST classification.'''
    def __init__(self, output_size=10):
        super(ResNet18, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_size)

    def forward(self, x):
        return self.model(x)

        