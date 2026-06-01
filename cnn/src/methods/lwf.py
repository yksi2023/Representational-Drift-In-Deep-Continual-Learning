import copy
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from src.methods.base import BaseContinualMethod


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

    def compute_loss(self, outputs, labels, active_range, task_idx, inputs):
        """Task loss + lambda * distillation loss from the frozen teacher."""
        task_loss = self._task_loss(outputs, labels, active_range)

        old_classes = task_idx * self.increment  # all classes seen so far
        use_distill = (task_idx > 0) and (self.teacher_model is not None) and (old_classes > 0)
        if use_distill:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
            distill_loss = self._compute_distill_loss(outputs, teacher_outputs, old_classes)
        else:
            distill_loss = outputs.new_tensor(0.0)

        loss = task_loss + self.lwf_lambda * distill_loss
        return loss, {"Task Loss": task_loss.item(), "Distill Loss": distill_loss.item()}

    def after_task(self, task_idx: int, train_loader) -> None:
        """Snapshot current model as the teacher for the next task."""
        self.teacher_model = copy.deepcopy(self.model).to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
