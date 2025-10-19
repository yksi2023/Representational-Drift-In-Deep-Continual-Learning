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


class BasicBlock_Tiny(nn.Module):
    '''A basic residual block for ResNet adapted to Tiny ImageNet.'''
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_Tiny, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        identity = self.shortcut(x)
        out += identity
        out = self.final_relu(out)
        return out

class ResNet18_Tiny(nn.Module):
    '''ResNet-18 architecture adapted for Tiny ImageNet classification.'''
    def __init__(self, num_classes=200):
        super(ResNet18_Tiny, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_layer=nn.Sequential(
            BasicBlock_Tiny(64, 64),
            BasicBlock_Tiny(64, 64),
            BasicBlock_Tiny(64, 128, stride=2),
            BasicBlock_Tiny(128, 128),
            BasicBlock_Tiny(128, 256, stride=2),
            BasicBlock_Tiny(256, 256),
            BasicBlock_Tiny(256, 512, stride=2),
            BasicBlock_Tiny(512, 512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.conv_layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x