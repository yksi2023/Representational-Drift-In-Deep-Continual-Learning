import torch
from torch import nn
from torch.nn import functional as F
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


# ---------------------------------------------------------------------------
# CIFAR-stem ResNet18, trained from scratch with Group Norm + Weight
# Standardization + Zero-gamma residual init. Designed to be comparable
# in normalization style to Google BiT so drift across backbones is
# apples-to-apples.
# ---------------------------------------------------------------------------


class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization (Qiao et al. 2019).

    Standardizes the conv weight to zero-mean / unit-std per output channel
    on every forward pass. Combined with Group Norm, closes the gap to BN
    on vision tasks (used in Google BiT).
    """
    def forward(self, x):
        w = self.weight
        w_mean = w.mean(dim=(1, 2, 3), keepdim=True)
        # Use reshape (not view) because channels_last makes the weight
        # non-contiguous in a way .view cannot handle.
        w_std = w.reshape(w.size(0), -1).std(dim=1, unbiased=False).reshape(-1, 1, 1, 1)
        w = (w - w_mean) / (w_std + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def _gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Group norm with group count auto-adjusted so it divides channels."""
    g = max_groups
    while num_channels % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, num_channels)


class BasicBlockGN(nn.Module):
    """ResNet basic block using StdConv2d + GroupNorm.

    The second GN's gamma is initialized to 0 by the parent model so the
    block starts as identity (``Zero-gamma``, a.k.a. "BN-0" / Fixup trick,
    He et al. 2019 Bag of Tricks). Helps early training stability when BN
    is removed.
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = StdConv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(out_c)
        self.conv2 = StdConv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = _gn(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                StdConv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                _gn(out_c),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu2(out)


class ResNet18CIFAR_GN(nn.Module):
    """ResNet-18 with CIFAR stem (3x3 stride 1, no maxpool) + GN + WS.

    Feature stages mirror torchvision naming (``layer1..layer4``) so drift
    probe specs transfer cleanly between this and other ResNet variants.
    """
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.stem = nn.Sequential(
            StdConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            _gn(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlockGN(64, 64),          BasicBlockGN(64, 64))
        self.layer2 = nn.Sequential(BasicBlockGN(64, 128, 2),      BasicBlockGN(128, 128))
        self.layer3 = nn.Sequential(BasicBlockGN(128, 256, 2),     BasicBlockGN(256, 256))
        self.layer4 = nn.Sequential(BasicBlockGN(256, 512, 2),     BasicBlockGN(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()
        self._zero_init_last_gn()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _zero_init_last_gn(self):
        """Zero-init gn2 gamma so each residual block starts as identity."""
        for m in self.modules():
            if isinstance(m, BasicBlockGN):
                nn.init.zeros_(m.gn2.weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Google BiT (ResNetV2-50x1, GN + WS, ImageNet-21k pretrained) via timm.
# ---------------------------------------------------------------------------


class BiTResNet50(nn.Module):
    """Google BiT-S ResNetV2-50x1 wrapper.

    Loads ``resnetv2_50x1_bit`` from timm (GN + WS + pre-activation blocks),
    swaps the classifier head, and supports coarse freezing.

    Freezable groups follow timm's module layout:
    ``stem, stages.0, stages.1, stages.2, stages.3, head``.
    """
    _layer_order = ["stem", "stages.0", "stages.1", "stages.2", "stages.3", "head"]

    def __init__(self, num_classes: int, pretrained: bool = True,
                 freeze_until=None, model_name: str = "resnetv2_50x1_bit"):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for BiTResNet50. Install via 'pip install timm'."
            ) from e

        # timm >=0.9 uses tagged model names; fall back to the known good tag.
        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                              num_classes=num_classes)
        except RuntimeError:
            self.backbone = timm.create_model(f"{model_name}.goog_in21k",
                                              pretrained=pretrained,
                                              num_classes=num_classes)

        self._frozen_names = set()
        if freeze_until:
            if freeze_until not in self._layer_order:
                raise ValueError(
                    f"freeze_until must be one of {self._layer_order}, got {freeze_until}")
            idx = self._layer_order.index(freeze_until)
            self._frozen_names = set(self._layer_order[:idx + 1])
            for name in self._frozen_names:
                mod = self._resolve_module(name)
                for p in mod.parameters():
                    p.requires_grad = False

    def _resolve_module(self, path: str) -> nn.Module:
        mod = self.backbone
        for part in path.split("."):
            if part.isdigit():
                mod = mod[int(part)]
            else:
                mod = getattr(mod, part)
        return mod

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for name in self._frozen_names:
                self._resolve_module(name).eval()
        return self

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


MODEL_DEFAULTS = {
    "fashion_mnist": {"model": "mlp",               "num_classes": 10,  "img_size": 28},
    "tiny_imagenet": {"model": "resnet18_pretrained", "num_classes": 200, "img_size": 224},
    "cifar100":      {"model": "resnet18_cifar_gn", "num_classes": 100, "img_size": 32},
}

MODEL_CHOICES = ("mlp", "resnet18_tiny", "resnet18_pretrained",
                 "resnet18_cifar_gn", "bit_r50x1")


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    """Instantiate a model by registry name."""
    name = name.lower()
    if name in ("mlp", "fashion_mnist_mlp"):
        return FashionMNISTModel(output_size=num_classes)
    if name == "resnet18_tiny":
        return ResNet18_Tiny(num_classes=num_classes)
    if name == "resnet18_pretrained":
        return PretrainedResNet18(
            num_classes=num_classes,
            pretrained=kwargs.get("pretrained", True),
            freeze_layers=kwargs.get("freeze_layers", ""),
            freeze_until=kwargs.get("freeze_until"),
        )
    if name == "resnet18_cifar_gn":
        return ResNet18CIFAR_GN(num_classes=num_classes)
    if name == "bit_r50x1":
        return BiTResNet50(
            num_classes=num_classes,
            pretrained=kwargs.get("pretrained", True),
            freeze_until=kwargs.get("freeze_until"),
        )
    raise ValueError(f"Unknown model '{name}'. Valid: {MODEL_CHOICES}")