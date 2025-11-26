import torch
from src.train import normal_train, replay_train
from src.utils import add_heads, EarlyStopping
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
     scheduler_config=None,
     batch_size=64, 
     val_loader=None, 
     method='normal',
     memory_size=5000, 
     save_dir: str = "experiments/ckpts", 
     run_comprehensive_eval=True,
     early_stopping_patience: int = 5,
     early_stopping_min_delta: float = 0.0,
     use_validation: bool = True,
     use_amp: bool = False,
     compile_model: bool = False,
     channels_last: bool = False):
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
    # Backend/runtime optimizations
    if device.type == 'cuda':
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    if channels_last:
        try:
            model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception:
            print("torch.compile not available or failed; continuing without compilation.")

    # Training parameters for checkpoint saving
    training_params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "method": method,
        "num_classes": num_classes,
        "increment": increment,
        "memory_size": memory_size if method == "replay" else None,
    }
    
    online_results = []

    if method.lower() == "normal":
        for i in range(0, num_classes, increment):
            print(f"\n{'='*60}")
            print(f"TASK {i//increment + 1}: Training on classes {list(range(i, i+increment))}")
            print(f"{'='*60}")
            
            train_loader = experiment_dataset.get_loader(mode='train', label=range(i, i+increment), batch_size=batch_size)
            test_loader = experiment_dataset.get_loader(mode='test', label=range(i, i+increment), batch_size=batch_size)
            current_val_loader = None
            if use_validation:
                try:
                    current_val_loader = experiment_dataset.get_loader(mode='val', label=range(i, i+increment), batch_size=batch_size)
                except Exception:
                    current_val_loader = None
            
            # Create a new scheduler for this task
            task_scheduler = None
            if scheduler_config is not None:
                if scheduler_config['type'] == 'ReduceLROnPlateau':
                    task_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode=scheduler_config.get('mode', 'min'),
                        factor=scheduler_config.get('factor', 0.5),
                        patience=scheduler_config.get('patience', 3),
                        min_lr=scheduler_config.get('min_lr', 1e-4)
                    )
            
            # Train the model (with optional validation and early stopping)
            early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)
            normal_train(model, train_loader, criterion, optimizer, device, epochs, val_loader=current_val_loader, early_stopping=early_stopper, scheduler=task_scheduler, use_amp=use_amp)

            # Save comprehensive checkpoint
            task_idx = i//increment + 1
            save_training_checkpoint(
                model, save_dir, task_idx, training_params,
                extra_metadata={
                    "classes": f'{i}-{i+increment-1}',
                }
            )
            
            # Evaluate on current task
            print(f"\nEvaluating Task {task_idx}:")
            ls, acc = evaluate(model, test_loader, criterion, device)
            online_results.append(acc)

    elif method.lower() == "replay":
        # Initialize persistent memory for replay method
        memory_set = {"data": [], "labels": []}
        
        for i in range(0, num_classes, increment):
            print(f"\n{'='*60}")
            print(f"TASK {i//increment + 1}: Training on classes {list(range(i, i+increment))}")
            print(f"Memory size: {len(memory_set['data'])} samples")
            print(f"{'='*60}")
            
            train_set = experiment_dataset.get_set(mode='train',label=range(i, i+increment))
            test_loader = experiment_dataset.get_loader(mode='test',label=range(i, i+increment), batch_size=batch_size)
            current_val_loader = None
            if use_validation:
                try:
                    current_val_loader = experiment_dataset.get_loader(mode='val', label=range(i, i+increment), batch_size=batch_size)
                except Exception:
                    current_val_loader = None
            
            # Create a new scheduler for this task
            task_scheduler = None
            if scheduler_config is not None:
                if scheduler_config['type'] == 'ReduceLROnPlateau':
                    task_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode=scheduler_config.get('mode', 'min'),
                        factor=scheduler_config.get('factor', 0.5),
                        patience=scheduler_config.get('patience', 3),
                        min_lr=scheduler_config.get('min_lr', 1e-4)
                    )
            
            # Update memory_set with the returned value from replay_train
            early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)
            memory_set = replay_train(model, train_set, criterion, optimizer, device, epochs, memory_set, memory_size, batch_size, val_loader=current_val_loader, early_stopping=early_stopper, scheduler=task_scheduler, use_amp=use_amp)
    
            # Save comprehensive checkpoint
            task_idx = i//increment + 1
            save_training_checkpoint(
                model, save_dir, task_idx, training_params,
                extra_metadata={
                    "classes": f'{i}-{i+increment-1}',
                }
            )
            
            # Evaluate on current task
            print(f"\nEvaluating Task {task_idx}:")
            ls, acc = evaluate(model, test_loader, criterion, device)
            online_results.append(acc)

    elif method.lower() == "ewc":
        # Placeholder for EWC method implementation
        pass
    
    # Run comprehensive evaluation after all tasks are completed
    if run_comprehensive_eval:
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE EVALUATION")
        print(f"{'='*60}")
        comprehensive_evaluation(
            model, online_results, experiment_dataset, device, num_classes, increment, criterion, save_dir
        )

