import torch
import tqdm
from src.methods.base import BaseContinualMethod
from src.eval import evaluate
from src.utils import EarlyStopping


class NormalMethod(BaseContinualMethod):
    """Standard fine-tuning without any continual learning mechanism."""
    
    def train_task(self, task_idx: int, train_loader, val_loader) -> None:
        """Train on a single task using standard fine-tuning."""
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
        
        # Restore best weights
        if val_loader is not None:
            early_stopper.restore(self.model)
