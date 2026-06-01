import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from src.methods.base import BaseContinualMethod


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
                    proj = grad_2d @ basis @ basis.T
                    grad_2d = grad_2d - proj
                    module.weight.grad.data = grad_2d.reshape(original_shape)
    
    def on_after_backward(self) -> None:
        """Project the accumulated gradients onto the orthogonal complement
        of the stored GPM feature space before the optimizer step."""
        self._project_gradient()
    
    def after_task(self, task_idx: int, train_loader) -> None:
        """Update GPM memory after each task."""
        if self.first_task_only_memory and task_idx > 0:
            print("Skipping GPM memory update (first_task_only_memory=True)")
            return
        
        print("Computing representation matrix for GPM...")
        rep_dict = self._get_representation_matrix(train_loader)
        self.gpm_memory = self._update_gpm_memory(rep_dict)
        
        if self.gpm_memory:
            for name in self.gpm_memory:
                self.gpm_memory[name] = self.gpm_memory[name].to(self.device)
            total_dims = sum(v.size(1) for v in self.gpm_memory.values())
            print(f"GPM memory updated: {len(self.gpm_memory)} layers, {total_dims} total basis vectors")
    
    def _get_representation_matrix(self, data_loader) -> Dict[str, torch.Tensor]:
        """Collect representation matrices from each layer."""
        self.model.eval()
        
        activations = {}
        hooks = []
        layer_map = {}
        
        def get_activation(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                inp = input[0] if isinstance(input, tuple) else input
                activations[name].append(inp.detach().cpu())
            return hook
        
        # In TIL mode, skip the classifier head (each task uses a different
        # output slice; projecting its gradients would unnecessarily constrain
        # new-task head learning).
        classifier_name = 'fc'  # all CNN models use self.fc as the classifier
        skip_names = {classifier_name} if self.learning_mode == 'til' else set()

        for name, module in self.model.named_modules():
            if name in skip_names:
                continue
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                layer_map[name] = module
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
            module = layer_map[name]
            
            if isinstance(module, torch.nn.Conv2d):
                # Unfold to (N, C*k*k, L)
                try:
                    unfolded = F.unfold(
                        act, 
                        kernel_size=module.kernel_size,
                        padding=module.padding,
                        stride=module.stride,
                        dilation=module.dilation
                    )
                    # unfolded: (N, C*k*k, L) -> (N, L, C*k*k) -> (N*L, C*k*k)
                    act = unfolded.transpose(1, 2).reshape(-1, unfolded.size(1))
                    
                    # Randomly subsample if too large (optional, but good for memory)
                    if act.size(0) > 100000:  # Heuristic threshold
                        indices = torch.randperm(act.size(0))[:100000]
                        act = act[indices]
                        
                except Exception as e:
                    print(f"Error unfolding {name}: {e}")
                    continue
                    
            elif isinstance(module, torch.nn.Linear):
                if act.dim() > 2:
                    act = act.reshape(-1, act.size(-1))
            
            rep_dict[name] = act
        
        self.model.train()
        return rep_dict
    
    def _update_gpm_memory(self, rep_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Update GPM memory using SVD."""
        updated_memory = {}
        
        for name, rep_matrix in rep_dict.items():
            rep_matrix = rep_matrix.float()
            
            # Record original energy before projecting out old subspace
            original_energy = (rep_matrix ** 2).sum().item()

            if self.gpm_memory is not None and name in self.gpm_memory:
                existing_basis = self.gpm_memory[name].cpu()
                proj = rep_matrix @ existing_basis @ existing_basis.T
                existing_energy = (proj ** 2).sum().item()
                residual = rep_matrix - proj

                # If old basis already covers enough, skip SVD entirely
                if existing_energy >= self.gpm_threshold * original_energy:
                    updated_memory[name] = existing_basis
                    continue
            else:
                residual = rep_matrix
                existing_energy = 0.0

            try:
                # No mean-centering: SVD on raw representations (matches GPM paper)
                U, S, Vh = torch.linalg.svd(residual, full_matrices=False)

                # Select k so that (existing_energy + new component energy)
                # >= threshold * original_energy
                needed_energy = self.gpm_threshold * original_energy - existing_energy
                cumsum = torch.cumsum(S ** 2, dim=0)
                k = (cumsum < needed_energy).sum().item() + 1
                k = max(1, min(k, S.size(0), residual.size(1)))

                new_basis = Vh[:k].T

                if self.gpm_memory is not None and name in self.gpm_memory:
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
