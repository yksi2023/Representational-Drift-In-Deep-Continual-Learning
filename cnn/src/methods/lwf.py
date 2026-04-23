import copy
import torch
import tqdm
import torch.nn.functional as F
from typing import Dict, Any, Optional
from src.methods.base import BaseContinualMethod
from src.eval import evaluate
from src.utils import EarlyStopping


class LwFMethod(BaseContinualMethod):
    """Learning without Forgetting via logit distillation from previous model."""

    def __init__(self, *args, lwf_lambda: float = 1.0, lwf_temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lwf_lambda = lwf_lambda
        self.lwf_temperature = lwf_temperature
        self.teacher_model = None

    def get_training_params(self) -> Dict[str, Any]:
        params = super().get_training_params()
        params["lwf_lambda"] = self.lwf_lambda
        params["lwf_temperature"] = self.lwf_temperature
        return params

    def _print_task_info(self, task_idx: int) -> None:
        if task_idx > 0:
            print(f"LwF lambda: {self.lwf_lambda}, temperature: {self.lwf_temperature}")

    def _get_extra_metadata(self) -> Optional[Dict]:
        return {
            "lwf_lambda": self.lwf_lambda,
            "lwf_temperature": self.lwf_temperature,
        }

    def _compute_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        old_classes: int,
    ) -> torch.Tensor:
        if old_classes <= 0:
            return student_logits.new_tensor(0.0)

        temperature = self.lwf_temperature
        student_log_probs = F.log_softmax(student_logits[:, :old_classes] / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits[:, :old_classes] / temperature, dim=1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    def train_task(self, task_idx: int, train_loader, val_loader) -> None:
        """Train with task loss + distillation loss from frozen previous model."""
        scheduler = self.create_scheduler()
        early_stopper = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        active_range = self.get_active_classes_range(task_idx)

        use_cuda_amp = bool(self.use_amp and (self.device.type == "cuda"))
        scaler = torch.amp.GradScaler() if use_cuda_amp else None

        old_classes = task_idx * self.increment  # all classes seen so far
        use_distill = (task_idx > 0) and (self.teacher_model is not None) and (old_classes > 0)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_task_loss = 0.0
            running_distill_loss = 0.0
            progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", disable=True)

            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs) if use_distill else None

                if use_cuda_amp:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(inputs)
                        if active_range is not None:
                            start_cls, end_cls = active_range
                            task_loss = self.criterion(outputs[:, start_cls:end_cls], labels - start_cls)
                        else:
                            task_loss = self.criterion(outputs, labels)

                        distill_loss = (
                            self._compute_distill_loss(outputs, teacher_outputs, old_classes)
                            if use_distill
                            else outputs.new_tensor(0.0)
                        )
                        loss = task_loss + self.lwf_lambda * distill_loss

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

                    distill_loss = (
                        self._compute_distill_loss(outputs, teacher_outputs, old_classes)
                        if use_distill
                        else outputs.new_tensor(0.0)
                    )
                    loss = task_loss + self.lwf_lambda * distill_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                running_loss += loss.item()
                running_task_loss += task_loss.item()
                running_distill_loss += distill_loss.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

            epoch_loss = running_loss / len(train_loader)
            epoch_task_loss = running_task_loss / len(train_loader)
            epoch_distill_loss = running_distill_loss / len(train_loader)
            print(
                f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {epoch_loss:.4f}, "
                f"Task Loss: {epoch_task_loss:.4f}, Distill Loss: {epoch_distill_loss:.6f}"
            )

            # Validation and early stopping
            val_loss = None
            if val_loader is not None:
                val_loss, val_acc = evaluate(
                    self.model,
                    val_loader,
                    self.criterion,
                    self.device,
                    active_classes_range=active_range,
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
        """Snapshot current model as the teacher for the next task."""
        self.teacher_model = copy.deepcopy(self.model).to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
