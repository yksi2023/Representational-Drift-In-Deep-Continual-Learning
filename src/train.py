import tqdm
from src.eval import evaluate
from src.utils import add_heads, update_memory, EarlyStopping
import torch
import torch.nn.functional as F
import random
import os

def normal_train(model, train_loader, criterion, optimizer, device, epochs, val_loader=None, early_stopping: EarlyStopping = None, scheduler=None, use_amp: bool = False, active_classes_range=None):
    use_cuda_amp = bool(use_amp and (device.type == 'cuda'))
    scaler = torch.amp.GradScaler() if use_cuda_amp else None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}",disable=True)
        for inputs, labels in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_cuda_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
                    if active_classes_range is not None:
                        start_cls, end_cls = active_classes_range
                        loss = criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                    else:
                        loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # unscale the gradients to the original value
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if active_classes_range is not None:
                    start_cls, end_cls = active_classes_range
                    loss = criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Validation and early stopping
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, active_classes_range=active_classes_range)
            print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Update learning rate scheduler based on validation loss
            if scheduler is not None:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current LR: {current_lr:.6f}")
            
            if early_stopping is not None:
                should_stop = early_stopping.step(model, val_loss)
                if should_stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    # Optionally restore best weights
    if val_loader is not None and early_stopping is not None:
        early_stopping.restore(model)


def replay_train(model, train_set, criterion, optimizer, device, epochs, memory_set, memory_size, batch_size=64, val_loader=None, early_stopping: EarlyStopping = None, scheduler=None, use_amp: bool = False, first_task_only_memory: bool = False, is_first_task: bool = True, active_classes_range=None):
    model.train()

    use_cuda_amp = bool(use_amp and (device.type == 'cuda'))
    scaler = torch.amp.GradScaler() if use_cuda_amp else None

    # dataloader perf defaults
    use_cuda = torch.cuda.is_available()
    cpu_count = os.cpu_count() or 2
    num_workers = max(2, min(8, cpu_count // 2))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs.update({
            "num_workers": num_workers,
            "persistent_workers": True,
            "prefetch_factor": 2,
        })

    # Handle empty memory case
    if len(memory_set["data"]) == 0:
        print("No memory samples available, training only on current task data")
        combined_loader = torch.utils.data.DataLoader(train_set, **loader_kwargs)
    else:
        # Create memory dataset - ensure all data tensors have the same shape
        try:
            # Stack memory data and convert labels to tensor
            memory_data_tensor = torch.stack(memory_set["data"])
            memory_labels_tensor = torch.tensor(memory_set["labels"], dtype=torch.long)

            # Create a custom dataset class that matches the format of train_set
            class MemoryDataset(torch.utils.data.Dataset):
                def __init__(self, data_tensor, labels_tensor):
                    self.data = data_tensor
                    self.labels = labels_tensor

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx], self.labels[idx].item()  # Return as tuple like train_set

            memory_dataset = MemoryDataset(memory_data_tensor, memory_labels_tensor)
            combined_dataset = torch.utils.data.ConcatDataset([train_set, memory_dataset])
            combined_loader = torch.utils.data.DataLoader(combined_dataset, **loader_kwargs)
        except Exception as e:
            print(f"Error creating memory dataset: {e}")
            print(f"Memory data shapes: {[d.shape for d in memory_set['data'][:5]]}")
            print(f"Memory labels: {memory_set['labels'][:10]}")
            # Fallback to training only on current task
            combined_loader = torch.utils.data.DataLoader(train_set, **loader_kwargs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm.tqdm(combined_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=True)
        for inputs, labels in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_cuda_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
                    if active_classes_range is not None:
                        start_cls, end_cls = active_classes_range
                        loss = criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                    else:
                        loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # unscale the gradients to the original value
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if active_classes_range is not None:
                    start_cls, end_cls = active_classes_range
                    loss = criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
        epoch_loss = running_loss / len(combined_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Validation and early stopping
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, active_classes_range=active_classes_range)
            print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Update learning rate scheduler based on validation loss
            if scheduler is not None:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current LR: {current_lr:.6f}")
            
            if early_stopping is not None:
                should_stop = early_stopping.step(model, val_loss)
                if should_stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    # Optionally restore best weights for replay
    if val_loader is not None and early_stopping is not None:
        early_stopping.restore(model)

    # Update memory set after training
    # If first_task_only_memory is True, only update memory for the first task
    if first_task_only_memory and not is_first_task:
        print(f"first_task_only_memory enabled: skipping memory update for non-first task")
        updated_memory = memory_set
    else:
        new_data = []
        new_labels = []

        # Sample from current task data to add to memory
        # Single pass: group indices by class
        class_to_indices = {}
        
        # Optimization: Try to access targets directly to avoid IO bottleneck (critical for ImageFolder)
        targets = None
        # Check if it's a Subset and we can access the underlying targets efficiently
        if isinstance(train_set, torch.utils.data.Subset) and hasattr(train_set.dataset, 'targets'):
             try:
                 full_targets = train_set.dataset.targets
                 indices = train_set.indices
                 # Ensure indices are handled correctly whether list or tensor
                 if torch.is_tensor(indices):
                     indices = indices.tolist()
                 
                 # Access targets
                 if torch.is_tensor(full_targets):
                     targets = full_targets[indices].tolist()
                 elif isinstance(full_targets, list):
                     targets = [int(full_targets[i]) for i in indices]
             except Exception as e:
                 # If anything goes wrong (e.g. index out of bounds, incompatible types), fallback
                 targets = None
        
        if targets is not None:
            # Fast path: use pre-fetched targets
            for i, label in enumerate(targets):
                if label not in class_to_indices:
                    class_to_indices[label] = []
                class_to_indices[label].append(i)
        else:
            # Slow path: iterate dataset (reads images from disk for ImageFolder)
            for i, (_, label) in enumerate(train_set):
                label = int(label)
                if label not in class_to_indices:
                    class_to_indices[label] = []
                class_to_indices[label].append(i)
        
        current_task_classes = set(class_to_indices.keys())
        seen_labels_set = set(memory_set["labels"]) if len(memory_set["labels"]) > 0 else set()
        total_classes = len(seen_labels_set.union(current_task_classes)) if len(current_task_classes) > 0 else max(1, len(seen_labels_set))
        samples_per_class = max(1, memory_size // max(1, total_classes))

        # Sample data from current task using precomputed indices
        for class_label, indices in class_to_indices.items():
            num_samples = min(samples_per_class, len(indices))
            selected_indices = random.sample(indices, num_samples)

            for idx in selected_indices:
                data, label = train_set[idx]
                # Ensure data is a tensor and label is an integer
                if isinstance(data, torch.Tensor):
                    new_data.append(data.cpu())
                else:
                    new_data.append(torch.tensor(data).cpu())
                new_labels.append(int(label))

        # Update memory with new samples
        updated_memory = update_memory(memory_set, new_data, new_labels, memory_size)
        print(f"Memory updated: {len(updated_memory['data'])} total samples")

    # Debug information

    return updated_memory

def compute_fisher_information(model, data_loader, criterion, device, num_samples=None):
    """
    Compute the Fisher Information Matrix (diagonal approximation) for EWC.
    
    Args:
        model: The trained model
        data_loader: DataLoader for the task data
        criterion: Loss function
        device: Device to run on
        num_samples: Number of samples to use (None = use all)
    
    Returns:
        fisher_dict: Dictionary mapping parameter names to their Fisher information values
        optimal_params: Dictionary mapping parameter names to their optimal values
    """
    model.eval()
    fisher_dict = {}
    optimal_params = {}
    
    # Initialize Fisher information to zero
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
            optimal_params[name] = param.data.clone()
    
    # Accumulate gradients squared
    sample_count = 0
    for inputs, labels in data_loader:
        if num_samples is not None and sample_count >= num_samples:
            break
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.detach() ** 2
        
        sample_count += inputs.size(0)
    
    # Average the Fisher information
    total_samples = sample_count if num_samples is None else min(sample_count, num_samples)
    for name in fisher_dict:
        fisher_dict[name] /= max(1, total_samples)
    
    model.train()
    return fisher_dict, optimal_params



def ewc_train(model, train_loader, criterion, optimizer, device, epochs, 
              fisher_dict=None, optimal_params=None, ewc_lambda=1000.0,
              val_loader=None, early_stopping: EarlyStopping = None, 
              scheduler=None, use_amp: bool = False, active_classes_range=None):
    """
    Train with EWC regularization.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for current task
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        epochs: Number of training epochs
        fisher_dict: Fisher information from first task (None for first task)
        optimal_params: Optimal parameters from first task (None for first task)
        ewc_lambda: EWC regularization strength
        val_loader: Validation DataLoader
        early_stopping: EarlyStopping instance
        scheduler: Learning rate scheduler
        use_amp: Whether to use automatic mixed precision
        active_classes_range: Tuple of (start_class, end_class) to mask the output
    """
    use_cuda_amp = bool(use_amp and (device.type == 'cuda'))
    scaler = torch.amp.GradScaler() if use_cuda_amp else None
    
    def compute_ewc_penalty():
        """Compute the EWC penalty term."""
        if fisher_dict is None or optimal_params is None:
            return 0.0
        
        penalty = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in fisher_dict:
                penalty += (fisher_dict[name] * (param - optimal_params[name]) ** 2).sum()
        return penalty

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_task_loss = 0.0
        running_ewc_loss = 0.0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=True)
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_cuda_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
                    if active_classes_range is not None:
                        start_cls, end_cls = active_classes_range
                        task_loss = criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                    else:
                        task_loss = criterion(outputs, labels)
                # EWC penalty computed outside autocast (full precision)
                ewc_penalty = compute_ewc_penalty()
                loss = task_loss + (ewc_lambda / 2.0) * ewc_penalty
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if active_classes_range is not None:
                    start_cls, end_cls = active_classes_range
                    task_loss = criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                else:
                    task_loss = criterion(outputs, labels)
                ewc_penalty = compute_ewc_penalty()
                loss = task_loss + (ewc_lambda / 2.0) * ewc_penalty
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            running_loss += loss.item()
            running_task_loss += task_loss.item()
            if isinstance(ewc_penalty, float):
                running_ewc_loss += ewc_penalty
            else:
                running_ewc_loss += ewc_penalty.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
        
        epoch_loss = running_loss / len(train_loader)
        epoch_task_loss = running_task_loss / len(train_loader)
        epoch_ewc_loss = running_ewc_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {epoch_loss:.4f}, Task Loss: {epoch_task_loss:.4f}, EWC Loss: {epoch_ewc_loss:.6f}")

        # Validation and early stopping
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, active_classes_range=active_classes_range)
            print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            if scheduler is not None:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current LR: {current_lr:.6f}")
            
            if early_stopping is not None:
                should_stop = early_stopping.step(model, val_loss)
                if should_stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    # Optionally restore best weights
    if val_loader is not None and early_stopping is not None:
        early_stopping.restore(model)
