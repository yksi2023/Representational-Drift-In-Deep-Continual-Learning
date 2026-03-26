import torch
import tqdm
from typing import Dict, Any, Optional
from src.methods.base import BaseContinualMethod
from src.eval import evaluate
from src.utils import EarlyStopping


class EWCMethod(BaseContinualMethod):
    """Elastic Weight Consolidation: penalize changes to important parameters."""
    
    def __init__(self, *args, ewc_lambda: float = 1000.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = None
        self.optimal_params = None
    
    def get_training_params(self) -> Dict[str, Any]:
        params = super().get_training_params()
        params["ewc_lambda"] = self.ewc_lambda
        return params
    
    def _print_task_info(self, task_idx: int) -> None:
        if task_idx > 0:
            print(f"EWC lambda: {self.ewc_lambda}")
    
    def _get_extra_metadata(self) -> Optional[Dict]:
        return {"ewc_lambda": self.ewc_lambda}
    
    def _compute_ewc_penalty(self) -> float:
        """Compute the EWC penalty term."""
        if self.fisher_dict is None or self.optimal_params is None:
            return 0.0
        
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_dict:
                penalty += (self.fisher_dict[name] * (param - self.optimal_params[name]) ** 2).sum()
        return penalty
    
    def train_task(self, task_idx: int, train_loader, val_loader) -> None:
        """Train with EWC regularization."""
        scheduler = self.create_scheduler()
        early_stopper = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta
        )
        active_range = self.get_active_classes_range(task_idx)
        
        use_cuda_amp = bool(self.use_amp and (self.device.type == 'cuda'))
        scaler = torch.amp.GradScaler() if use_cuda_amp else None
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_task_loss = 0.0
            running_ewc_loss = 0.0
            progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", disable=True)
            
            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                if use_cuda_amp:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(inputs)
                        if active_range is not None:
                            start_cls, end_cls = active_range
                            task_loss = self.criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                        else:
                            task_loss = self.criterion(outputs, labels)
                    ewc_penalty = self._compute_ewc_penalty()
                    loss = task_loss + (self.ewc_lambda / 2.0) * ewc_penalty
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(inputs)
                    if active_range is not None:
                        start_cls, end_cls = active_range
                        task_loss = self.criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                    else:
                        task_loss = self.criterion(outputs, labels)
                    ewc_penalty = self._compute_ewc_penalty()
                    loss = task_loss + (self.ewc_lambda / 2.0) * ewc_penalty
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()
                
                running_loss += loss.item()
                running_task_loss += task_loss.item()
                running_ewc_loss += ewc_penalty if isinstance(ewc_penalty, float) else ewc_penalty.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
            
            epoch_loss = running_loss / len(train_loader)
            epoch_task_loss = running_task_loss / len(train_loader)
            epoch_ewc_loss = running_ewc_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {epoch_loss:.4f}, "
                  f"Task Loss: {epoch_task_loss:.4f}, EWC Loss: {epoch_ewc_loss:.6f}")
            
            # Validation and early stopping
            if val_loader is not None:
                val_loss, val_acc = evaluate(
                    self.model, val_loader, self.criterion, self.device,
                    active_classes_range=active_range
                )
                print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
                
                if scheduler is not None:
                    scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Current LR: {current_lr:.6f}")
                
                should_stop = early_stopper.step(self.model, val_loss)
                if should_stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        if val_loader is not None:
            early_stopper.restore(self.model)
    
    def after_task(self, task_idx: int, train_loader) -> None:
        """Compute Fisher information after first task."""
        if task_idx == 0:
            print("Computing Fisher Information Matrix for first task...")
            self.fisher_dict, self.optimal_params = self._compute_fisher_information(train_loader)
            print(f"Fisher information computed for {len(self.fisher_dict)} parameters")
    
    def _compute_fisher_information(self, data_loader, num_samples=None):
        """Compute diagonal Fisher Information Matrix for EWC."""
        self.model.eval()
        fisher_dict = {}
        optimal_params = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)
                optimal_params[name] = param.data.clone()
        
        sample_count = 0
        for inputs, labels in data_loader:
            if num_samples is not None and sample_count >= num_samples:
                break
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.detach() ** 2
            
            sample_count += inputs.size(0)
        
        total_samples = sample_count if num_samples is None else min(sample_count, num_samples)
        for name in fisher_dict:
            fisher_dict[name] /= max(1, total_samples)
        
        self.model.train()
        return fisher_dict, optimal_params
