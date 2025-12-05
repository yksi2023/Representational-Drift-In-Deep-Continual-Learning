import tqdm
from src.eval import evaluate
from src.utils import add_heads, update_memory, EarlyStopping
import torch
import random
import os

def normal_train(model, train_loader, criterion, optimizer, device, epochs, val_loader=None, early_stopping: EarlyStopping = None, scheduler=None, use_amp: bool = False):
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
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # unscale the gradients to the original value
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
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
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
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


def replay_train(model, train_set, criterion, optimizer, device, epochs, memory_set, memory_size, batch_size=64, val_loader=None, early_stopping: EarlyStopping = None, scheduler=None, use_amp: bool = False):
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
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # unscale the gradients to the original value
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
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
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
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
    new_data = []
    new_labels = []

    # Sample from current task data to add to memory
    # Calculate how many samples to add per class in current task (projected class-balanced quota)
    current_task_classes = set()
    for _, label in train_set:
        current_task_classes.add(label)

    seen_labels_set = set(memory_set["labels"]) if len(memory_set["labels"]) > 0 else set()
    total_classes = len(seen_labels_set.union(current_task_classes)) if len(current_task_classes) > 0 else max(1, len(seen_labels_set))
    samples_per_class = max(1, memory_size // max(1, total_classes))

    # Sample data from current task
    for class_label in current_task_classes:
        class_indices = [i for i, (_, label) in enumerate(train_set) if label == class_label]
        if len(class_indices) > 0:
            # Sample up to samples_per_class from this class
            num_samples = min(samples_per_class, len(class_indices))
            selected_indices = random.sample(class_indices, num_samples)

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


def replay_train_deprecated(model, train_set, criterion, optimizer, device, epochs, memory_set, memory_size, batch_size=64, val_loader=None, early_stopping: EarlyStopping = None, scheduler=None, use_amp: bool = False):
    model.train()
    '''neglect this function'''
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

    # 计算当前任务类别数
    current_task_classes = set()
    for _, label in train_set:
        current_task_classes.add(label)
    num_current_classes = len(current_task_classes)

    # 计算 memory 类别数
    memory_classes = set(memory_set["labels"]) if len(memory_set["labels"]) > 0 else set()
    num_memory_classes = len(memory_classes)

    # 创建当前任务的 DataLoader
    current_loader = torch.utils.data.DataLoader(train_set, **loader_kwargs)

    # 创建 memory 的 DataLoader（如果有 memory 数据）
    memory_loader = None
    if len(memory_set["data"]) > 0:
        try:
            memory_data_tensor = torch.stack(memory_set["data"])
            memory_labels_tensor = torch.tensor(memory_set["labels"], dtype=torch.long)

            class MemoryDataset(torch.utils.data.Dataset):
                def __init__(self, data_tensor, labels_tensor):
                    self.data = data_tensor
                    self.labels = labels_tensor

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx], self.labels[idx].item()

            memory_dataset = MemoryDataset(memory_data_tensor, memory_labels_tensor)
            memory_loader = torch.utils.data.DataLoader(memory_dataset, **loader_kwargs)
        except Exception as e:
            print(f"Error creating memory dataset: {e}")
            memory_loader = None

    # 计算权重（按类别数比例）
    if memory_loader is not None and num_memory_classes > 0:
        total_classes = num_current_classes + num_memory_classes
        weight_current = num_current_classes / total_classes
        weight_memory = num_memory_classes / total_classes
    else:
        weight_current = 1.0
        weight_memory = 0.0
        print("No memory samples available, training only on current task data")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_current_loss = 0.0
        running_memory_loss = 0.0

        # 创建迭代器
        current_iter = iter(current_loader)
        memory_iter = iter(memory_loader) if memory_loader is not None else None

        num_batches = len(current_loader)
        progress_bar = tqdm.tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", disable=True)

        for _ in progress_bar:
            
            optimizer.zero_grad(set_to_none=True)

            # 获取当前任务 batch
            try:
                current_inputs, current_labels = next(current_iter)
            except StopIteration:
                current_iter = iter(current_loader)
                current_inputs, current_labels = next(current_iter)

            current_inputs = current_inputs.to(device, non_blocking=True)
            current_labels = current_labels.to(device, non_blocking=True)

            # 获取 memory batch（如果有）
            memory_inputs, memory_labels = None, None
            if memory_iter is not None:
                try:
                    memory_inputs, memory_labels = next(memory_iter)
                except StopIteration:
                    memory_iter = iter(memory_loader)
                    memory_inputs, memory_labels = next(memory_iter)
                memory_inputs = memory_inputs.to(device, non_blocking=True)
                memory_labels = memory_labels.to(device, non_blocking=True)

            if use_cuda_amp:
                with torch.amp.autocast(device_type=device.type):
                    # 计算当前任务 loss
                    current_outputs = model(current_inputs)
                    current_loss = criterion(current_outputs, current_labels)

                    # 计算 memory loss（如果有）
                    if memory_inputs is not None:
                        memory_outputs = model(memory_inputs)
                        memory_loss = criterion(memory_outputs, memory_labels)
                        # 加权总 loss
                        total_loss = weight_current * current_loss + weight_memory * memory_loss
                    else:
                        memory_loss = torch.tensor(0.0)
                        total_loss = current_loss

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 计算当前任务 loss
                current_outputs = model(current_inputs)
                current_loss = criterion(current_outputs, current_labels)

                # 计算 memory loss（如果有）
                if memory_inputs is not None:
                    memory_outputs = model(memory_inputs)
                    memory_loss = criterion(memory_outputs, memory_labels)
                    # 加权总 loss
                    total_loss = weight_current * current_loss + weight_memory * memory_loss
                else:
                    memory_loss = torch.tensor(0.0)
                    total_loss = current_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            running_loss += total_loss.item()
            running_current_loss += current_loss.item()
            running_memory_loss += memory_loss.item() if memory_inputs is not None else 0.0
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        epoch_loss = running_loss / num_batches
        epoch_current_loss = running_current_loss / num_batches
        epoch_memory_loss = running_memory_loss / num_batches if memory_loader else 0.0
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {epoch_loss:.4f}, Current: {epoch_current_loss:.4f}, Memory: {epoch_memory_loss:.4f}")

        # Validation and early stopping
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
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

    # Optionally restore best weights for replay
    if val_loader is not None and early_stopping is not None:
        early_stopping.restore(model)

    # Update memory set after training (保持原有逻辑不变)
    new_data = []
    new_labels = []

    seen_labels_set = set(memory_set["labels"]) if len(memory_set["labels"]) > 0 else set()
    total_classes = len(seen_labels_set.union(current_task_classes)) if len(current_task_classes) > 0 else max(1, len(seen_labels_set))
    samples_per_class = max(1, memory_size // max(1, total_classes))

    for class_label in current_task_classes:
        class_indices = [i for i, (_, label) in enumerate(train_set) if label == class_label]
        if len(class_indices) > 0:
            num_samples = min(samples_per_class, len(class_indices))
            selected_indices = random.sample(class_indices, num_samples)

            for idx in selected_indices:
                data, label = train_set[idx]
                if isinstance(data, torch.Tensor):
                    new_data.append(data.cpu())
                else:
                    new_data.append(torch.tensor(data).cpu())
                new_labels.append(int(label))

    updated_memory = update_memory(memory_set, new_data, new_labels, memory_size)
    print(f"Memory updated: {len(updated_memory['data'])} total samples")

    return updated_memory