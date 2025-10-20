import torch
from src.models import FashionMNISTModel
from src.continual import incremental_learning
from datasets import IncrementalFashionMNIST

increment=2
batch_size=64
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTModel(output_size=increment)
model.to(device)
epochs = 1
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
data_manager = IncrementalFashionMNIST()
task_classes = torch.bincount(data_manager.train_set.targets)

incremental_learning(model, data_manager, epochs, device, task_classes, increment, criterion, optimizer, batch_size=batch_size, val_loader=None, method="normal")

