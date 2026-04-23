"""Temporary script to print BiTResNet50_IN1k model structure."""

import sys
sys.path.insert(0, "cnn/src")
from models import BiTResNet50_IN1k

model = BiTResNet50_IN1k(num_classes=200, pretrained=False)
print(model)
