import torch
from src.train import normal_train, replay_train
from src.utils import add_heads
from src.eval import evaluate
import os
from src.checkpoints import save_model


def incremental_learning(model, experiment_dataset, epochs, device, num_classes, increment, criterion, optimizer, batch_size=64, val_loader=None, method='normal',memory_size=200, save_dir: str = "experiments/ckpts"):
    """
    Train the model incrementally on new tasks.

    Args:
        model: The neural network model to be trained.
        increment_dataset: The dataset for the new task.
        val_loader: DataLoader for validation data of the new task.
        epochs: Number of epochs to train.
        device: Device to run the training on (CPU or GPU).
        num_classes: Total number of classes in the dataset.
        increment: Number of new classes in the current task.
        criterion: Loss function for training.
        optimizer: Optimizer for training.
        method: Training method to use ('normal' or 'replay').
    """
    if method.lower() == "normal":
        for i in range(0, num_classes, increment):
            train_loader = experiment_dataset.get_loader(mode='train', label=range(i, i+increment))
            test_loader = experiment_dataset.get_loader(mode='test', label=range(i, i+increment))
            normal_train(model, train_loader, criterion, optimizer, device, epochs)

            # save model after each task
            task_idx = i//increment + 1
            save_model(model, save_dir, task_idx, extra_metadata={
                "method": method,
                "classes": list(range(i, i+increment)),
            })
            evaluate(model, test_loader, criterion, device)
            if i + increment < num_classes:
                model = add_heads(model, increment)

    if method.lower() == "replay":
        for i in range(0, num_classes, increment):
            train_set = experiment_dataset.get_set(mode='train',label=range(i, i+increment))
            test_set = experiment_dataset.get_set(mode='test',label=range(i, i+increment))
            
            memory_set = {"data":[], "labels":[]}
            replay_train(model, train_set, criterion, optimizer, device, epochs, memory_set, memory_size, batch_size)
    
            # save model after each task
            task_idx = i//increment + 1
            save_model(model, save_dir, task_idx, extra_metadata={
                "method": method,
                "classes": list(range(i, i+increment)),
            })
            evaluate(model, test_set, criterion, device)



    if method.lower() == "ewc":
        # Placeholder for EWC method implementation
        pass

