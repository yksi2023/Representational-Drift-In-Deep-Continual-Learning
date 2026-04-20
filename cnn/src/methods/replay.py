import torch
import tqdm
import random
import os
from typing import Dict, Any, Optional
from src.methods.base import BaseContinualMethod
from src.eval import evaluate
from src.utils import EarlyStopping, update_memory


class ReplayMethod(BaseContinualMethod):
    """Experience Replay: store and replay samples from previous tasks."""
    
    def __init__(self, *args, memory_size: int = 5000,
                 memory_per_class: Optional[int] = None,
                 first_task_only_memory: bool = False, **kwargs):
        """Replay buffer.

        Two allocation modes are supported:

        * ``memory_per_class`` (not None): iCaRL-style per-class exemplar
          memory. Stores exactly ``memory_per_class`` samples for every
          class seen so far; previously stored exemplars are never
          replaced. Total buffer grows linearly with classes seen.

        * Otherwise: fixed total budget ``memory_size`` shared evenly
          across seen classes. The per-class quota shrinks as more tasks
          arrive (older exemplars are subsampled to fit).
        """
        super().__init__(*args, **kwargs)
        self.memory_size = memory_size
        self.memory_per_class = memory_per_class
        self.first_task_only_memory = first_task_only_memory
        self.memory_set = {"data": [], "labels": []}

    def get_training_params(self) -> Dict[str, Any]:
        params = super().get_training_params()
        params["memory_size"] = self.memory_size
        params["memory_per_class"] = self.memory_per_class
        params["first_task_only_memory"] = self.first_task_only_memory
        return params
    
    def _print_task_info(self, task_idx: int) -> None:
        print(f"Memory size: {len(self.memory_set['data'])} samples")
    
    def get_active_classes_range(self, task_idx: int) -> Optional[tuple]:
        """For replay, active range spans all seen classes in TIL mode."""
        if self.learning_mode != "til":
            return None
        end_class = (task_idx + 1) * self.increment
        return (0, end_class)
    
    def train_task(self, task_idx: int, train_loader, val_loader) -> None:
        """Train using replay of stored samples."""
        # Get current task dataset (not loader) for combining with memory
        start_class = task_idx * self.increment
        end_class = start_class + self.increment
        train_set = self.experiment_dataset.get_set(mode='train', label=range(start_class, end_class))
        
        scheduler = self.create_scheduler()
        early_stopper = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta
        )
        active_range = self.get_active_classes_range(task_idx)
        
        # Create combined loader with memory
        combined_loader = self._create_combined_loader(train_set)
        
        use_cuda_amp = bool(self.use_amp and (self.device.type == 'cuda'))
        scaler = torch.amp.GradScaler() if use_cuda_amp else None
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm.tqdm(combined_loader, desc=f"Epoch {epoch+1}/{self.epochs}", disable=True)
            
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()
                
                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
            
            epoch_loss = running_loss / len(combined_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")
            
            # Validation and early stopping
            val_loss = None
            if val_loader is not None:
                val_loss, val_acc = evaluate(
                    self.model, val_loader, self.criterion, self.device,
                    active_classes_range=active_range
                )
                print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            self.step_scheduler(scheduler, val_loss)
            if scheduler is not None:
                print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if val_loader is not None and early_stopper.step(self.model, val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if val_loader is not None:
            early_stopper.restore(self.model)
    
    def after_task(self, task_idx: int, train_loader) -> None:
        """Update memory after training each task."""
        is_first_task = (task_idx == 0)
        
        if self.first_task_only_memory and not is_first_task:
            print("first_task_only_memory enabled: skipping memory update")
            return
        
        # Get current task dataset
        start_class = task_idx * self.increment
        end_class = start_class + self.increment
        train_set = self.experiment_dataset.get_set(mode='train', label=range(start_class, end_class))
        
        self._update_memory(train_set)
    
    def _create_combined_loader(self, train_set):
        """Create a DataLoader combining current task data with memory."""
        use_cuda = torch.cuda.is_available()
        cpu_count = os.cpu_count() or 2
        num_workers = max(2, min(8, cpu_count // 2))
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "pin_memory": use_cuda,
        }
        if num_workers > 0:
            loader_kwargs.update({
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 2,
            })
        
        if len(self.memory_set["data"]) == 0:
            print("No memory samples available, training only on current task data")
            return torch.utils.data.DataLoader(train_set, **loader_kwargs)
        
        try:
            memory_data_tensor = torch.stack(self.memory_set["data"])
            memory_labels_tensor = torch.tensor(self.memory_set["labels"], dtype=torch.long)
            
            class MemoryDataset(torch.utils.data.Dataset):
                def __init__(self, data_tensor, labels_tensor):
                    self.data = data_tensor
                    self.labels = labels_tensor
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx], self.labels[idx].item()
            
            memory_dataset = MemoryDataset(memory_data_tensor, memory_labels_tensor)
            combined_dataset = torch.utils.data.ConcatDataset([train_set, memory_dataset])
            return torch.utils.data.DataLoader(combined_dataset, **loader_kwargs)
        except Exception as e:
            print(f"Error creating memory dataset: {e}")
            return torch.utils.data.DataLoader(train_set, **loader_kwargs)
    
    def _update_memory(self, train_set):
        """Sample from current task and update memory."""
        new_data = []
        new_labels = []
        
        # Build class to indices mapping
        class_to_indices = {}
        targets = None
        
        # Walk through (possibly nested) Subsets down to the base dataset's
        # targets, composing the index mapping at each level. Without this,
        # ImageNet-21k-P (which wraps ImageFolder in Subset twice) would fall
        # through to the slow per-sample label read below.
        try:
            ds = train_set
            sub_indices = None
            while isinstance(ds, torch.utils.data.Subset):
                layer_idx = ds.indices
                if torch.is_tensor(layer_idx):
                    layer_idx = layer_idx.tolist()
                if sub_indices is None:
                    sub_indices = list(layer_idx)
                else:
                    sub_indices = [layer_idx[i] for i in sub_indices]
                ds = ds.dataset
            if hasattr(ds, 'targets') and sub_indices is not None:
                full_targets = ds.targets
                if torch.is_tensor(full_targets):
                    targets = full_targets[sub_indices].tolist()
                else:
                    targets = [int(full_targets[i]) for i in sub_indices]
        except Exception:
            targets = None
        
        if targets is not None:
            for i, label in enumerate(targets):
                if label not in class_to_indices:
                    class_to_indices[label] = []
                class_to_indices[label].append(i)
        else:
            for i, (_, label) in enumerate(train_set):
                label = int(label)
                if label not in class_to_indices:
                    class_to_indices[label] = []
                class_to_indices[label].append(i)
        
        current_task_classes = set(class_to_indices.keys())

        # Decide per-class quota for the current task's classes.
        if self.memory_per_class is not None:
            samples_per_class = self.memory_per_class
        else:
            seen_labels_set = set(self.memory_set["labels"]) if self.memory_set["labels"] else set()
            total_classes = len(seen_labels_set.union(current_task_classes)) or 1
            samples_per_class = max(1, self.memory_size // total_classes)

        for class_label, indices in class_to_indices.items():
            num_samples = min(samples_per_class, len(indices))
            selected_indices = random.sample(indices, num_samples)

            for idx in selected_indices:
                data, label = train_set[idx]
                if isinstance(data, torch.Tensor):
                    new_data.append(data.cpu())
                else:
                    new_data.append(torch.tensor(data).cpu())
                new_labels.append(int(label))

        if self.memory_per_class is not None:
            # iCaRL-style: append new exemplars, leave older ones untouched.
            self.memory_set["data"].extend(new_data)
            self.memory_set["labels"].extend(new_labels)
        else:
            # Total-budget mode: merge + class-balanced prune to memory_size.
            self.memory_set = update_memory(self.memory_set, new_data, new_labels, self.memory_size)

        classes_in_memory = len(set(self.memory_set["labels"]))
        print(f"Memory updated: {len(self.memory_set['data'])} samples "
              f"across {classes_in_memory} classes "
              f"({samples_per_class}/class this task)")
