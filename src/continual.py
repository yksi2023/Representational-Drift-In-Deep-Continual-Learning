import torch
from src.train import normal_train, replay_train
from src.utils import add_heads
from src.eval import evaluate
import os


def incremental_learning(model, experiment_dataset, epochs, device, task_classes, increment, criterion, optimizer, batch_size=64, val_loader=None, method='normal',memory_size=200):
    """
    Train the model incrementally on new tasks.

    Args:
        model: The neural network model to be trained.
        increment_dataset: The dataset for the new task.
        val_loader: DataLoader for validation data of the new task.
        epochs: Number of epochs to train.
        device: Device to run the training on (CPU or GPU).
        task_classes: List of classes in the current task.
        increment: Number of new classes in the current task.
        criterion: Loss function for training.
        optimizer: Optimizer for training.
        method: Training method to use ('normal' or 'replay').
    """
    if method.lower() == "normal":
        for i in range(0, len(task_classes), increment):
            train_loader = experiment_dataset.get_loader(mode='train', label=range(i, i+increment))
            test_loader = experiment_dataset.get_loader(mode='test', label=range(i, i+increment))
            normal_train(model, train_loader, criterion, optimizer, device, epochs)

            # save model after each task
            save_dir = "experiments/fashion_mnist_incremental"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/model_after_task_{i//increment + 1}.pth")
            evaluate(model, test_loader, criterion, device)
            if i + increment < len(task_classes):
                model = add_heads(model, increment)

    if method.lower() == "replay":
        for i in range(0, len(task_classes), increment):
            train_set = experiment_dataset.get_set(mode='train',label=range(i, i+increment))
            test_set = experiment_dataset.get_set(mode='test',label=range(i, i+increment))
            
            memory_set = {"data":[], "labels":[]}
            replay_train(model, train_set, criterion, optimizer, device, epochs, memory_set, memory_size, batch_size)
    
            # save model after each task
            save_dir = "experiments/Tiny_ImageNet_incremental"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/model_after_task_{i//increment + 1}.pth")
            evaluate(model, test_set, criterion, device)



    if method.lower() == "ewc":
        # Placeholder for EWC method implementation
        pass

