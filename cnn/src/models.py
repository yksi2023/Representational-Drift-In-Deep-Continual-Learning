import torch
from torch import nn
from torchvision import models as tv_models

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


class PretrainedResNet18(nn.Module):
    '''Torchvision ResNet18 pretrained on ImageNet with configurable layer freezing.'''
    def __init__(self, num_classes, pretrained = True, freeze_layers=None, freeze_until = None):
        super().__init__()
        # Load backbone
        try:
            # Torchvision >= 0.13 uses weights API
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet18(weights=weights)
        except Exception:
            # Fallback for older versions
            backbone = tv_models.resnet18(pretrained=pretrained)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

        # Determine layers order for convenience
        self._layer_order = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']

        # Normalize freeze configuration
        freeze_set = set()
        if freeze_layers:
            if isinstance(freeze_layers, (list, tuple, set)):
                freeze_set.update(str(x) for x in freeze_layers)
            elif isinstance(freeze_layers, str) and freeze_layers.strip():
                # support comma-separated
                parts = [p.strip() for p in freeze_layers.split(',') if p.strip()]
                freeze_set.update(parts)
        if freeze_until:
            if freeze_until not in self._layer_order:
                raise ValueError(f"freeze_until must be one of {self._layer_order}, got {freeze_until}")
            idx = self._layer_order.index(freeze_until)
            for name in self._layer_order[:idx+1]:
                freeze_set.add(name)

        # Apply freezing
        for name, module in self.backbone.named_children():
            if name in freeze_set:
                for p in module.parameters():
                    p.requires_grad = False

        # Optionally put frozen BatchNorms in eval mode during training (user can still override externally)
        self._frozen_names = freeze_set

    def train(self, mode = True):
        super().train(mode)
        # Keep BN layers inside frozen modules in eval mode
        if mode and self._frozen_names:
            for name, module in self.backbone.named_children():
                if name in self._frozen_names:
                    module.eval()
        return self

    def forward(self, x):
        return self.backbone(x)