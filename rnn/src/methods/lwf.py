import copy
import torch
import torch.nn.functional as F
from src.methods.base import BaseMethod
from src.train import compute_loss


class LwFMethod(BaseMethod):
    """Learning without Forgetting: distill knowledge from a frozen teacher model."""

    def __init__(self, lwf_lambda=1.0, lwf_temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.lwf_lambda = lwf_lambda
        self.lwf_temperature = lwf_temperature
        self.teacher_model = None

    def _distillation_loss(self, student_logits, teacher_logits, mask):
        """KL divergence distillation loss, masked over time steps."""
        T = self.lwf_temperature
        # student_logits, teacher_logits: (seq_len, batch, output_size)
        s_log_probs = F.log_softmax(student_logits / T, dim=-1)
        t_probs = F.softmax(teacher_logits / T, dim=-1)
        # KL per timestep per batch: sum over output dim
        kl = F.kl_div(s_log_probs, t_probs, reduction='none').sum(dim=-1)  # (seq, batch)
        masked_kl = kl * mask
        return masked_kl.mean() * (T ** 2)

    def train_step(self, optimizer, trial):
        """Training step with task loss + distillation loss from teacher."""
        self.model.train()
        optimizer.zero_grad()

        x, y, mask = trial.to_tensor(device=self.device)
        outputs = self.model(x, return_all_states=False)
        task_loss = compute_loss(outputs, y, mask,
                                 loss_type=trial.config.get('loss_type', 'cross_entropy'))

        loss = task_loss
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(x, return_all_states=False)
            distill_loss = self._distillation_loss(outputs, teacher_outputs, mask)
            loss = task_loss + self.lwf_lambda * distill_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    def after_task(self, task_idx, task_name, task_gen_fn):
        """Snapshot current model as teacher for future tasks."""
        print(f"  Snapshotting model as teacher after {task_name}")
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
