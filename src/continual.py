import torch
from src.train import normal_train, replay_train
from src.utils import add_heads
from src.eval import evaluate, comprehensive_evaluation
import os
from src.checkpoints import save_training_checkpoint


def incremental_learning(model,
     experiment_dataset, 
     epochs, device, 
     num_classes, 
     increment, 
     criterion, 
     optimizer, 
     batch_size=64, 
     val_loader=None, 
     method='normal',
     memory_size=5000, 
     save_dir: str = "experiments/ckpts", 
     run_comprehensive_eval=True):
    """
    Train the model incrementally on new tasks.

    Args:
        model: The neural network model to be trained.
        experiment_dataset: The dataset for the new task.
        val_loader: DataLoader for validation data of the new task.
        epochs: Number of epochs to train.
        device: Device to run the training on (CPU or GPU).
        num_classes: Total number of classes in the dataset.
        increment: Number of new classes in the current task.
        criterion: Loss function for training.
        optimizer: Optimizer for training.
        method: Training method to use ('normal' or 'replay').
        run_comprehensive_eval: Whether to run comprehensive evaluation after all tasks.
    """
    # Training parameters for checkpoint saving
    training_params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "method": method,
        "num_classes": num_classes,
        "increment": increment,
        "memory_size": memory_size if method == "replay" else None,
    }
    
    if method.lower() == "normal":
        for i in range(0, num_classes, increment):
            print(f"\n{'='*60}")
            print(f"TASK {i//increment + 1}: Training on classes {list(range(i, i+increment))}")
            print(f"{'='*60}")
            
            train_loader = experiment_dataset.get_loader(mode='train', label=range(i, i+increment))
            test_loader = experiment_dataset.get_loader(mode='test', label=range(i, i+increment))
            
            # Train the model
            normal_train(model, train_loader, criterion, optimizer, device, epochs)

            # Save comprehensive checkpoint
            task_idx = i//increment + 1
            save_training_checkpoint(
                model, optimizer, save_dir, task_idx, epochs, training_params,
                extra_metadata={
                    "method": method,
                    "classes": list(range(i, i+increment)),
                    "task_description": f"Task {task_idx} - Classes {list(range(i, i+increment))}"
                }
            )
            
            # Evaluate on current task
            print(f"\nEvaluating Task {task_idx}:")
            evaluate(model, test_loader, criterion, device)

    elif method.lower() == "replay":
        # Initialize persistent memory for replay method
        memory_set = {"data": [], "labels": []}
        
        for i in range(0, num_classes, increment):
            print(f"\n{'='*60}")
            print(f"TASK {i//increment + 1}: Training on classes {list(range(i, i+increment))}")
            print(f"Memory size: {len(memory_set['data'])} samples")
            print(f"{'='*60}")
            
            train_set = experiment_dataset.get_set(mode='train',label=range(i, i+increment))
            test_loader = experiment_dataset.get_loader(mode='test',label=range(i, i+increment))
            
            # Update memory_set with the returned value from replay_train
            memory_set = replay_train(model, train_set, criterion, optimizer, device, epochs, memory_set, memory_size, batch_size)
    
            # Save comprehensive checkpoint
            task_idx = i//increment + 1
            save_training_checkpoint(
                model, optimizer, save_dir, task_idx, epochs, training_params,
                extra_metadata={
                    "method": method,
                    "classes": list(range(i, i+increment)),
                    "task_description": f"Task {task_idx} - Classes {list(range(i, i+increment))}",
                    "memory_size": memory_size
                }
            )
            
            # Evaluate on current task
            print(f"\nEvaluating Task {task_idx}:")
            evaluate(model, test_loader, criterion, device)

    elif method.lower() == "ewc":
        # Placeholder for EWC method implementation
        pass
    
    # Run comprehensive evaluation after all tasks are completed
    if run_comprehensive_eval:
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE EVALUATION")
        print(f"{'='*60}")
        comprehensive_evaluation(
            model, experiment_dataset, device, num_classes, increment, criterion, save_dir
        )

