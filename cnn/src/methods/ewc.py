import torch
from typing import Dict, Any, Optional
from src.methods.base import BaseContinualMethod


class EWCMethod(BaseContinualMethod):
    """Elastic Weight Consolidation: penalize changes to important parameters.
    
    Args:
        ewc_lambda: Strength of the EWC penalty.
        ewc_protect: Which tasks to protect.
            'first' — only compute Fisher for the first task (default).
            'all'   — compute Fisher after every task and accumulate penalties.
    """
    
    def __init__(self, *args, ewc_lambda: float = 1000.0, ewc_protect: str = 'first', **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.ewc_protect = ewc_protect
        # 'first' mode: single (fisher, θ*) pair
        self.fisher_dict = None
        self.optimal_params = None
        # 'all' mode: list of (fisher_dict, optimal_params), one per completed task
        self.ewc_tasks = []
    
    def get_training_params(self) -> Dict[str, Any]:
        params = super().get_training_params()
        params["ewc_lambda"] = self.ewc_lambda
        params["ewc_protect"] = self.ewc_protect
        return params
    
    def _print_task_info(self, task_idx: int) -> None:
        if task_idx > 0:
            print(f"EWC lambda: {self.ewc_lambda}")
    
    def _get_extra_metadata(self) -> Optional[Dict]:
        return {"ewc_lambda": self.ewc_lambda, "ewc_protect": self.ewc_protect}
    
    def _compute_ewc_penalty(self) -> float:
        """Compute the EWC penalty term."""
        penalty = 0.0
        if self.ewc_protect == 'all' and self.ewc_tasks:
            for fisher_dict, optimal_params in self.ewc_tasks:
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in fisher_dict:
                        penalty += (fisher_dict[name] * (param - optimal_params[name]) ** 2).sum()
        elif self.fisher_dict is not None and self.optimal_params is not None:
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.fisher_dict:
                    penalty += (self.fisher_dict[name] * (param - self.optimal_params[name]) ** 2).sum()
        return penalty
    
    def compute_loss(self, outputs, labels, active_range, task_idx, inputs):
        """Task loss + (lambda/2) * EWC penalty."""
        task_loss = self._task_loss(outputs, labels, active_range)
        ewc_penalty = self._compute_ewc_penalty()
        loss = task_loss + (self.ewc_lambda / 2.0) * ewc_penalty
        penalty_val = ewc_penalty if isinstance(ewc_penalty, float) else ewc_penalty.item()
        return loss, {"Task Loss": task_loss.item(), "EWC Loss": penalty_val}
    
    def after_task(self, task_idx: int, train_loader) -> None:
        """Compute Fisher information after task."""
        if self.ewc_protect == 'first':
            if task_idx == 0:
                print("Computing Fisher Information Matrix for first task...")
                self.fisher_dict, self.optimal_params = self._compute_fisher_information(train_loader, task_idx)
                print(f"Fisher information computed for {len(self.fisher_dict)} parameters")
        else:  # 'all'
            print(f"Computing Fisher Information Matrix for task {task_idx + 1} "
                  f"(total EWC terms: {len(self.ewc_tasks) + 1})...")
            fisher, optpar = self._compute_fisher_information(train_loader, task_idx)
            self.ewc_tasks.append((fisher, optpar))
            print(f"Fisher information computed for {len(fisher)} parameters")
    
    def _compute_fisher_information(self, data_loader, task_idx: int, num_samples=None):
        """Compute diagonal Fisher Information Matrix for EWC."""
        self.model.eval()
        fisher_dict = {}
        optimal_params = {}

        active_range = self.get_active_classes_range(task_idx)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)
                optimal_params[name] = param.data.clone()

        batch_count = 0
        sample_count = 0
        for inputs, labels in data_loader:
            if num_samples is not None and sample_count >= num_samples:
                break

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            if active_range is not None:
                start_cls, end_cls = active_range
                loss = self.criterion(outputs[:, start_cls:end_cls], labels - start_cls)
            else:
                loss = self.criterion(outputs, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.detach() ** 2

            batch_count += 1
            sample_count += inputs.size(0)

        for name in fisher_dict:
            fisher_dict[name] /= max(1, batch_count)

        self.model.train()
        return fisher_dict, optimal_params
