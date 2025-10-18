import torch
from src.train import normal_train
from src.utils import add_heads
from src.eval import evaluate
import os


def incremental_learning(model, increment_dataset, epochs, device, task_classes, increment, criterion, optimizer, val_loader=None):
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
    """
    for i in range(0, len(task_classes), increment):
        increment_dataset.next_task()
        train_loader = increment_dataset.get_set(mode='train')
        test_loader = increment_dataset.get_set(mode='test')
        normal_train(model, train_loader, criterion, optimizer, device, epochs)

        # save model after each task
        save_dir = "experiments/fashion_mnist_incremental"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/model_after_task_{i//increment + 1}.pth")
        evaluate(model, test_loader, criterion, device)
        if increment_dataset.current_class < len(task_classes):
            model = add_heads(model, increment)


def incremental_learning_with_replay(model, increment_dataset, epochs, device, task_classes, increment, criterion, optimizer, val_loader=None, memory=None):
    """
    Train the model incrementally on new tasks with experience replay.

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
        memory: A replay memory containing samples from previous tasks.
    """
    for i in range(0, len(task_classes), increment):
        train_loader = increment_dataset.get_set(mode='train')
        test_loader = increment_dataset.get_set(mode='test')
        normal_train(model, train_loader, criterion, optimizer, device, epochs, memory=memory)

        # save model after each task
        torch.save(model.state_dict(), f"model_after_task_{i//increment + 1}.pth")
        evaluate(model, test_loader, criterion, device)
        if increment_dataset.current_class < len(task_classes):
            model = add_heads(model, increment)