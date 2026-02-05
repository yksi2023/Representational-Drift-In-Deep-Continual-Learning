import torch
import tqdm
from typing import Dict, Any, Optional
from src.methods.base import BaseContinualMethod
from src.eval import evaluate
from src.utils import EarlyStopping


class GPMMethod(BaseContinualMethod):
    """Gradient Projection Memory: project gradients to preserve old task knowledge."""
    
    def __init__(self, *args, gpm_threshold: float = 0.99, gpm_num_samples: int = 300, 
                 first_task_only_memory: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpm_threshold = gpm_threshold
        self.gpm_num_samples = gpm_num_samples
        self.gpm_memory = None
        self.first_task_only_memory = first_task_only_memory
    
    def get_training_params(self) -> Dict[str, Any]:
        params = super().get_training_params()
        params["gpm_threshold"] = self.gpm_threshold
        params["first_task_only_memory"] = self.first_task_only_memory
        return params
    
    def _print_task_info(self, task_idx: int) -> None:
        if task_idx > 0:
            print(f"GPM threshold: {self.gpm_threshold}")
            if self.gpm_memory:
                total_dims = sum(v.size(1) for v in self.gpm_memory.values())
                print(f"GPM memory: {len(self.gpm_memory)} layers, {total_dims} total basis vectors")
    
    def _get_extra_metadata(self) -> Optional[Dict]:
        return {"gpm_threshold": self.gpm_threshold}
    
    def _project_gradient(self) -> None:
        """Project gradients onto orthogonal complement of GPM feature space."""
        if self.gpm_memory is None:
            return
        
        for name, module in self.model.named_modules():
            if name in self.gpm_memory and hasattr(module, 'weight') and module.weight.grad is not None:
                basis = self.gpm_memory[name]
                grad = module.weight.grad.data
                original_shape = grad.shape
                
                if isinstance(module, torch.nn.Conv2d):
                    grad_2d = grad.reshape(grad.size(0), -1)
                else:
                    grad_2d = grad
                
                if basis.size(0) == grad_2d.size(1):
                    device = grad.device
                    basis = basis.to(device)
                    proj = grad_2d @ basis @ basis.T
                    grad_2d = grad_2d - proj
                    module.weight.grad.data = grad_2d.reshape(original_shape)
    
    def train_task(self, task_idx: int, train_loader, val_loader) -> None:
        """Train with gradient projection."""
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
                            loss = self.criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                        else:
                            loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    
                    self._project_gradient()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(inputs)
                    if active_range is not None:
                        start_cls, end_cls = active_range
                        loss = self.criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                    else:
                        loss = self.criterion(outputs, labels)
                    loss.backward()
                    
                    self._project_gradient()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()
                
                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
            
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")
            
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
        """Update GPM memory after each task."""
        if self.first_task_only_memory and task_idx > 0:
            print("Skipping GPM memory update (first_task_only_memory=True)")
            return
        
        print("Computing representation matrix for GPM...")
        rep_dict = self._get_representation_matrix(train_loader)
        self.gpm_memory = self._update_gpm_memory(rep_dict)
        
        if self.gpm_memory:
            total_dims = sum(v.size(1) for v in self.gpm_memory.values())
            print(f"GPM memory updated: {len(self.gpm_memory)} layers, {total_dims} total basis vectors")
    
    def _get_representation_matrix(self, data_loader) -> Dict[str, torch.Tensor]:
        """Collect representation matrices from each layer."""
        self.model.eval()
        
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                inp = input[0] if isinstance(input, tuple) else input
                activations[name].append(inp.detach().cpu())
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
        
        sample_count = 0
        with torch.no_grad():
            for inputs, _ in data_loader:
                if sample_count >= self.gpm_num_samples:
                    break
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
                sample_count += inputs.size(0)
        
        for hook in hooks:
            hook.remove()
        
        rep_dict = {}
        for name, act_list in activations.items():
            act = torch.cat(act_list, dim=0)[:self.gpm_num_samples]
            
            if act.dim() == 4:  # Conv layer
                N, C, H, W = act.shape
                act = act.permute(0, 2, 3, 1).reshape(-1, C)
            elif act.dim() == 2:  # Linear layer
                pass
            else:
                continue
            
            rep_dict[name] = act
        
        self.model.train()
        return rep_dict
    
    def _update_gpm_memory(self, rep_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Update GPM memory using SVD."""
        updated_memory = {}
        
        for name, rep_matrix in rep_dict.items():
            rep_matrix = rep_matrix.float()
            
            if self.gpm_memory is not None and name in self.gpm_memory:
                existing_basis = self.gpm_memory[name]
                proj = rep_matrix @ existing_basis @ existing_basis.T
                rep_matrix = rep_matrix - proj
            
            try:
                mean = rep_matrix.mean(dim=0, keepdim=True)
                centered = rep_matrix - mean
                
                U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
                
                total_var = (S ** 2).sum()
                cumsum = torch.cumsum(S ** 2, dim=0)
                k = (cumsum < self.gpm_threshold * total_var).sum().item() + 1
                k = max(1, min(k, S.size(0), rep_matrix.size(1)))
                
                new_basis = Vh[:k].T
                
                if self.gpm_memory is not None and name in self.gpm_memory:
                    existing_basis = self.gpm_memory[name]
                    combined = torch.cat([existing_basis, new_basis], dim=1)
                    Q, R = torch.linalg.qr(combined)
                    diag = torch.abs(torch.diag(R))
                    keep = diag > 1e-6
                    updated_memory[name] = Q[:, keep]
                else:
                    updated_memory[name] = new_basis
                    
            except Exception as e:
                print(f"SVD failed for layer {name}: {e}")
                if self.gpm_memory is not None and name in self.gpm_memory:
                    updated_memory[name] = self.gpm_memory[name]
        
        return updated_memory
