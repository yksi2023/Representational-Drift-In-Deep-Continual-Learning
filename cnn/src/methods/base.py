import torch
import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from src.eval import evaluate, comprehensive_evaluation
from src.checkpoints import save_training_checkpoint
from src.utils import EarlyStopping


class BaseContinualMethod(ABC):
    """Base class for continual learning methods."""
    
    def __init__(
        self,
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
        save_dir: str = "experiments/ckpts",
        run_comprehensive_eval: bool = True,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.0,
        use_validation: bool = True,
        use_amp: bool = False,
        learning_mode: str = "til",
    ):
        self.model = model
        self.experiment_dataset = experiment_dataset
        self.epochs = epochs
        self.device = device
        self.num_classes = num_classes
        self.increment = increment
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.run_comprehensive_eval = run_comprehensive_eval
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.use_validation = use_validation
        self.use_amp = use_amp
        self.learning_mode = learning_mode
        
        # Derived
        self.num_tasks = num_classes // increment

        # Results tracking
        self.online_results = []         # diagonal: acc on task_k after training task_k
        self.first_task_results = []     # first row: acc on task_0 after training task_k
        # Full task x training-stage matrix in RNN-compatible format:
        # {task_idx: [ {loss, accuracy} after stage 0, stage 1, ... ]}
        self.performance_history: Dict[int, list] = {i: [] for i in range(self.num_tasks)}

        # Snapshot the initial LR of each param group so we can restore it
        # at the start of every task (schedulers mutate param_groups[*]['lr']).
        self._initial_lrs = [g['lr'] for g in self.optimizer.param_groups]
    
    @property
    def method_name(self) -> str:
        """Return the method name for logging."""
        return self.__class__.__name__.replace('Method', '').lower()
    
    def get_training_params(self) -> Dict[str, Any]:
        """Return training parameters for checkpoint saving."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "method": self.method_name,
            "num_classes": self.num_classes,
            "increment": self.increment,
        }
    
    def create_scheduler(self):
        """Create a learning rate scheduler for the current task.

        Returned object may be a ReduceLROnPlateau (stepped with val_loss)
        or CosineAnnealingLR (stepped once per epoch). The training loops
        check which kind via isinstance.
        """
        if self.scheduler_config is None:
            return None
        cfg = self.scheduler_config
        if cfg['type'] == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=cfg.get('mode', 'min'),
                factor=cfg.get('factor', 0.5),
                patience=cfg.get('patience', 3),
                min_lr=cfg.get('min_lr', 1e-4),
            )
        if cfg['type'] == 'CosineAnnealingLR':
            # Reset base_lrs to current LR so each new task starts the
            # cosine cycle from the full LR (optimizer.state was cleared).
            for g in self.optimizer.param_groups:
                g['initial_lr'] = g['lr']
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.get('T_max', self.epochs),
                eta_min=cfg.get('eta_min', 1e-4),
            )
        return None
    
    def step_scheduler(self, scheduler, val_loss: Optional[float] = None) -> None:
        """Step a scheduler with the right signature.

        - ReduceLROnPlateau: needs val_loss; silently skipped if None.
        - CosineAnnealingLR / other _LRScheduler: stepped once (per epoch).
        """
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                scheduler.step(val_loss)
        else:
            scheduler.step()

    def get_task_loaders(self, task_idx: int) -> Tuple:
        """Get train/test/val loaders for a task."""
        start_class = task_idx * self.increment
        end_class = start_class + self.increment
        class_range = range(start_class, end_class)
        
        train_loader = self.experiment_dataset.get_loader(
            mode='train', label=class_range, batch_size=self.batch_size
        )
        test_loader = self.experiment_dataset.get_loader(
            mode='test', label=class_range, batch_size=self.batch_size
        )
        
        val_loader = None
        if self.use_validation:
            try:
                val_loader = self.experiment_dataset.get_loader(
                    mode='val', label=class_range, batch_size=self.batch_size
                )
            except Exception:
                val_loader = None
        
        return train_loader, test_loader, val_loader
    
    def get_active_classes_range(self, task_idx: int) -> Optional[Tuple[int, int]]:
        """Get the active class range for TIL mode."""
        if self.learning_mode != "til":
            return None
        start_class = task_idx * self.increment
        end_class = start_class + self.increment
        return (start_class, end_class)
    
    def before_task(self, task_idx: int) -> None:
        """Hook called before training each task. Override in subclasses."""
        pass
    
    @abstractmethod
    def train_task(self, task_idx: int, train_loader, val_loader) -> None:
        """Train on a single task. Must be implemented by subclasses."""
        pass
    
    def after_task(self, task_idx: int, train_loader) -> None:
        """Hook called after training each task. Override in subclasses."""
        pass
    
    def evaluate_task(self, task_idx: int, test_loader) -> float:
        """Evaluate on current task and return accuracy."""
        active_range = self.get_active_classes_range(task_idx)
        print(f"\nEvaluating Task {task_idx + 1}:")
        _, acc = evaluate(
            self.model, test_loader, self.criterion, self.device,
            active_classes_range=active_range
        )
        return acc
    
    def evaluate_first_task(self) -> float:
        """Evaluate on the first task (forgetting measure)."""
        first_task_loader = self.experiment_dataset.get_loader(
            mode='test', label=range(0, self.increment), batch_size=self.batch_size
        )
        first_task_range = (0, self.increment) if self.learning_mode == "til" else None
        print(f"Evaluating on First Task (classes 0-{self.increment-1}):")
        _, acc = evaluate(
            self.model, first_task_loader, self.criterion, self.device,
            active_classes_range=first_task_range
        )
        return acc

    def _evaluate_and_record_all(self, current_task_idx: int) -> None:
        """Evaluate on every task (past, current, future) and append to performance_history.

        Mirrors the RNN's task x training-stage matrix so we can plot the same
        accuracy heatmap. Also populates legacy arrays used by downstream
        plotting (online / first-task).
        """
        self.model.eval()
        print(f"\n--- Evaluation after Task {current_task_idx + 1} ---")
        for eval_idx in range(self.num_tasks):
            start_cls = eval_idx * self.increment
            end_cls = start_cls + self.increment
            test_loader = self.experiment_dataset.get_loader(
                mode='test', label=range(start_cls, end_cls),
                batch_size=self.batch_size, shuffle=False,
            )
            task_range = (start_cls, end_cls) if self.learning_mode == "til" else None
            loss, acc_pct = evaluate(
                self.model, test_loader, self.criterion, self.device,
                active_classes_range=task_range,
            )
            # Store accuracy as fraction [0,1] for heatmap-friendly format.
            self.performance_history[eval_idx].append({
                "loss": float(loss),
                "accuracy": float(acc_pct) / 100.0,
            })
            tag = "*" if eval_idx == current_task_idx else " "
            print(f"  {tag} task_{eval_idx}: loss={loss:.4f}, acc={acc_pct:.2f}%")

        # Legacy aggregates used by the old line plots
        self.online_results.append(self.performance_history[current_task_idx][-1]["accuracy"] * 100.0)
        self.first_task_results.append(self.performance_history[0][-1]["accuracy"] * 100.0)
    
    def save_checkpoint(self, task_idx: int, extra_metadata: Optional[Dict] = None) -> None:
        """Save checkpoint after a task."""
        start_class = task_idx * self.increment
        end_class = start_class + self.increment
        metadata = {"classes": f'{start_class}-{end_class-1}'}
        if extra_metadata:
            metadata.update(extra_metadata)
        save_training_checkpoint(
            self.model, self.save_dir, task_idx + 1,
            self.get_training_params(), extra_metadata=metadata
        )
    
    def save_metrics(self) -> None:
        """Save training metrics to JSON."""
        metrics = {
            "online_results": self.online_results,
            "first_task_results": self.first_task_results,
        }
        metrics_path = os.path.join(self.save_dir, "training_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Training metrics saved to {metrics_path}")

        # Full accuracy matrix (task x training-stage), RNN-compatible format
        perf_dict = {f"task_{k}": v for k, v in self.performance_history.items()}
        perf_path = os.path.join(self.save_dir, "performance_history.json")
        with open(perf_path, "w", encoding="utf-8") as f:
            json.dump(perf_dict, f, ensure_ascii=False, indent=2)
        print(f"Performance history saved to {perf_path}")
    
    def run(self) -> None:
        """Main training loop over all tasks."""
        for task_idx in range(self.num_tasks):
            self.optimizer.state.clear()
            # Restore starting LR so schedulers begin each task from the top.
            for g, lr0 in zip(self.optimizer.param_groups, self._initial_lrs):
                g['lr'] = lr0
            start_class = task_idx * self.increment
            end_class = start_class + self.increment
            
            print(f"\n{'='*60}")
            print(f"TASK {task_idx + 1}: Training on classes {list(range(start_class, end_class))}")
            self._print_task_info(task_idx)
            print(f"{'='*60}")
            
            train_loader, test_loader, val_loader = self.get_task_loaders(task_idx)
            
            # Hooks and training
            self.before_task(task_idx)
            self.train_task(task_idx, train_loader, val_loader)
            self.after_task(task_idx, train_loader)
            
            # Checkpoint and full-matrix evaluation on every task
            self.save_checkpoint(task_idx, self._get_extra_metadata())
            self._evaluate_and_record_all(task_idx)
        
        # Final save and comprehensive evaluation
        self.save_metrics()
        
        if self.run_comprehensive_eval:
            print(f"\n{'='*60}")
            print("RUNNING COMPREHENSIVE EVALUATION")
            print(f"{'='*60}")
            comprehensive_evaluation(
                self.model, self.online_results, self.first_task_results,
                self.experiment_dataset, self.device, self.num_classes,
                self.increment, self.criterion,
                learning_mode=self.learning_mode,
                save_dir=self.save_dir,
            )
    
    def _print_task_info(self, task_idx: int) -> None:
        """Print additional task info. Override in subclasses."""
        pass
    
    def _get_extra_metadata(self) -> Optional[Dict]:
        """Get extra metadata for checkpoint. Override in subclasses."""
        return None
