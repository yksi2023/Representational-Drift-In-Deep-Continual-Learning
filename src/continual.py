import torch
from typing import Optional, Dict
from src.methods import get_method


def incremental_learning(
    model: torch.nn.Module,
    experiment_dataset,
    epochs: int,
    device: torch.device,
    num_classes: int,
    increment: int,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_config: Optional[Dict] = None,
    batch_size: int = 64,
    val_loader=None,  # Deprecated, kept for compatibility
    method: str = 'normal',
    memory_size: int = 5000,
    save_dir: str = "experiments/ckpts",
    run_comprehensive_eval: bool = True,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
    use_validation: bool = True,
    use_amp: bool = False,
    compile_model: bool = False,
    channels_last: bool = False,
    first_task_only_memory: bool = False,
    ewc_lambda: float = 1000.0,
    gpm_threshold: float = 0.99,
    learning_mode: str = "til",
):
    """
    Train the model incrementally on new tasks.

    Args:
        model: The neural network model to be trained.
        experiment_dataset: The dataset for the experiment.
        epochs: Number of epochs to train per task.
        device: Device to run the training on (CPU or GPU).
        num_classes: Total number of classes in the dataset.
        increment: Number of new classes in each task.
        criterion: Loss function for training.
        optimizer: Optimizer for training.
        scheduler_config: Configuration for learning rate scheduler.
        batch_size: Batch size for training.
        method: Training method ('normal', 'replay', 'ewc', 'gpm').
        memory_size: Size of memory buffer for replay method.
        save_dir: Directory to save checkpoints.
        run_comprehensive_eval: Whether to run comprehensive evaluation after all tasks.
        early_stopping_patience: Patience for early stopping.
        early_stopping_min_delta: Minimum delta for early stopping.
        use_validation: Whether to use validation set.
        use_amp: Whether to use automatic mixed precision.
        compile_model: Whether to use torch.compile.
        channels_last: Whether to use channels_last memory format.
        first_task_only_memory: For replay, only store memory from first task.
        ewc_lambda: Lambda for EWC regularization.
        gpm_threshold: Threshold for GPM feature space.
        learning_mode: 'til' for task-incremental or 'cil' for class-incremental.
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

    # Common kwargs for all methods
    common_kwargs = {
        "model": model,
        "experiment_dataset": experiment_dataset,
        "epochs": epochs,
        "device": device,
        "num_classes": num_classes,
        "increment": increment,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler_config": scheduler_config,
        "batch_size": batch_size,
        "save_dir": save_dir,
        "run_comprehensive_eval": run_comprehensive_eval,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "use_validation": use_validation,
        "use_amp": use_amp,
        "learning_mode": learning_mode,
    }

    # Method-specific kwargs
    method_kwargs = {}
    method_lower = method.lower()
    
    if method_lower == 'replay':
        method_kwargs = {
            "memory_size": memory_size,
            "first_task_only_memory": first_task_only_memory,
        }
    elif method_lower == 'ewc':
        method_kwargs = {
            "ewc_lambda": ewc_lambda,
        }
    elif method_lower == 'gpm':
        method_kwargs = {
            "gpm_threshold": gpm_threshold,
            "first_task_only_memory": first_task_only_memory,
        }

    # Get method class and instantiate
    method_cls = get_method(method)
    learner = method_cls(**common_kwargs, **method_kwargs)
    
    # Run training
    learner.run()
