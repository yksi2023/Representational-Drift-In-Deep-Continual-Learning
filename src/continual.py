import torch
import json
from src.train import normal_train, replay_train, ewc_train, compute_fisher_information
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
     channels_last: bool = False,
     first_task_only_memory: bool = False,
     ewc_lambda: float = 1000.0,
     learning_mode: str = "til"):
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
        learning_mode: 'til' for task-incremental (masked output) or 'cil' for class-incremental (full output).
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
        "first_task_only_memory": first_task_only_memory if method == "replay" else None,
    }
    
    online_results = []
    first_task_results = []  # Track performance on first task after each task

    if method.lower() == "normal":
        for i in range(0, num_classes, increment):
            optimizer.state.clear()
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
            # Determine active_classes_range based on learning_mode
            active_range = (i, i + increment) if learning_mode == "til" else None
            normal_train(model, train_loader, criterion, optimizer, device, epochs, val_loader=current_val_loader, early_stopping=early_stopper, scheduler=task_scheduler, use_amp=use_amp, active_classes_range=active_range)

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
            ls, acc = evaluate(model, test_loader, criterion, device, active_classes_range=active_range)
            online_results.append(acc)
            
            # Evaluate on first task (forgetting measure)
            first_task_test_loader = experiment_dataset.get_loader(mode='test', label=range(0, increment), batch_size=batch_size)
            print(f"Evaluating on First Task (classes 0-{increment-1}):")
            first_task_range = (0, increment) if learning_mode == "til" else None
            _, first_task_acc = evaluate(model, first_task_test_loader, criterion, device, active_classes_range=first_task_range)
            first_task_results.append(first_task_acc)

    elif method.lower() == "replay":
        # Initialize persistent memory for replay method
        memory_set = {"data": [], "labels": []}
        
        for i in range(0, num_classes, increment):
            optimizer.state.clear()
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
            is_first_task = (i == 0)
            # Determine active_classes_range based on learning_mode
            active_range = (0, i + increment) if learning_mode == "til" else None
            memory_set = replay_train(model, train_set, criterion, optimizer, device, epochs, memory_set, memory_size, batch_size, val_loader=current_val_loader, early_stopping=early_stopper, scheduler=task_scheduler, use_amp=use_amp, first_task_only_memory=first_task_only_memory, is_first_task=is_first_task, active_classes_range=active_range)
    
            # Save comprehensive checkpoint
            task_idx = i//increment + 1
            save_training_checkpoint(
                model, save_dir, task_idx, training_params,
                extra_metadata={
                    "classes": f'{i}-{i+increment-1}',
                }
            )
            
            # Evaluate on current task (use current task range for TIL)
            print(f"\nEvaluating Task {task_idx}:")
            current_task_range = (i, i + increment) if learning_mode == "til" else None
            ls, acc = evaluate(model, test_loader, criterion, device, active_classes_range=current_task_range)
            online_results.append(acc)
            
            # Evaluate on first task (forgetting measure)
            first_task_test_loader = experiment_dataset.get_loader(mode='test', label=range(0, increment), batch_size=batch_size)
            print(f"Evaluating on First Task (classes 0-{increment-1}):")
            first_task_range = (0, increment) if learning_mode == "til" else None
            _, first_task_acc = evaluate(model, first_task_test_loader, criterion, device, active_classes_range=first_task_range)
            first_task_results.append(first_task_acc)

    elif method.lower() == "ewc":
        # EWC: Only compute Fisher/optimal_params after first task, apply penalty on subsequent tasks
        fisher_dict = None
        optimal_params = None
        
        for i in range(0, num_classes, increment):
            optimizer.state.clear()
            task_idx = i // increment + 1
            is_first_task = (i == 0)
            
            print(f"\n{'='*60}")
            print(f"TASK {task_idx}: Training on classes {list(range(i, i+increment))}")
            if not is_first_task:
                print(f"EWC lambda: {ewc_lambda}")
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
            
            # Train with EWC
            early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)
            # Determine active_classes_range based on learning_mode
            active_range = (i, i + increment) if learning_mode == "til" else None
            ewc_train(
                model, train_loader, criterion, optimizer, device, epochs,
                fisher_dict=fisher_dict,
                optimal_params=optimal_params,
                ewc_lambda=ewc_lambda,
                val_loader=current_val_loader,
                early_stopping=early_stopper,
                scheduler=task_scheduler,
                use_amp=use_amp,
                active_classes_range=active_range
            )
            
            # After first task, compute Fisher information and store optimal parameters
            if is_first_task:
                print("Computing Fisher Information Matrix for first task...")
                first_task_train_loader = experiment_dataset.get_loader(mode='train', label=range(0, increment), batch_size=batch_size)
                fisher_dict, optimal_params = compute_fisher_information(
                    model, first_task_train_loader, criterion, device
                )
                print(f"Fisher information computed for {len(fisher_dict)} parameters")
            
            # Save comprehensive checkpoint
            save_training_checkpoint(
                model, save_dir, task_idx, training_params,
                extra_metadata={
                    "classes": f'{i}-{i+increment-1}',
                    "ewc_lambda": ewc_lambda,
                }
            )
            
            # Evaluate on current task
            print(f"\nEvaluating Task {task_idx}:")
            ls, acc = evaluate(model, test_loader, criterion, device, active_classes_range=active_range)
            online_results.append(acc)
            
            # Evaluate on first task (forgetting measure)
            first_task_test_loader = experiment_dataset.get_loader(mode='test', label=range(0, increment), batch_size=batch_size)
            print(f"Evaluating on First Task (classes 0-{increment-1}):")
            first_task_range = (0, increment) if learning_mode == "til" else None
            _, first_task_acc = evaluate(model, first_task_test_loader, criterion, device, active_classes_range=first_task_range)
            first_task_results.append(first_task_acc)
    
    # Save training metrics for later analysis/plotting
    training_metrics = {
        "online_results": online_results,
        "first_task_results": first_task_results,
    }
    metrics_path = os.path.join(save_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, ensure_ascii=False, indent=2)
    print(f"Training metrics saved to {metrics_path}")
    
    # Run comprehensive evaluation after all tasks are completed
    if run_comprehensive_eval:
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE EVALUATION")
        print(f"{'='*60}")
        comprehensive_evaluation(
            model, online_results, first_task_results, experiment_dataset, device, num_classes, increment, criterion, save_dir,
           
        )

