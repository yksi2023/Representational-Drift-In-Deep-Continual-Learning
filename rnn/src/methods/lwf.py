import copy
import torch
import torch.nn.functional as F
from src.methods.base import BaseMethod
from src.train import compute_loss


class LwFMethod(BaseMethod):
    """Learning without Forgetting: distill knowledge from all past teacher models."""

    def __init__(self, lwf_lambda=1.0, lwf_temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.lwf_lambda = lwf_lambda
        self.lwf_temperature = lwf_temperature
        self.teachers = []  # list of frozen teacher models, one per completed task

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
        """Training step with task loss + distillation loss from all teachers."""
        self.model.train()
        optimizer.zero_grad()

        x, y, mask = trial.to_tensor(device=self.device)
        outputs = self.model(x, return_all_states=False)
        task_loss = compute_loss(outputs, y, mask,
                                 loss_type=trial.config.get('loss_type', 'cross_entropy'))

        loss = task_loss
        if self.teachers:
            distill_total = torch.tensor(0.0, device=self.device)
            with torch.no_grad():
                for teacher in self.teachers:
                    teacher_outputs = teacher(x, return_all_states=False)
                    distill_total = distill_total + self._distillation_loss(outputs, teacher_outputs, mask)
            loss = task_loss + self.lwf_lambda * distill_total / len(self.teachers)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    def after_task(self, task_idx, task_name, task_gen_fn):
        """Snapshot current model as teacher for future tasks."""
        print(f"  Snapshotting model as teacher after {task_name} (total teachers: {task_idx + 1})")
        teacher = copy.deepcopy(self.model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        self.teachers.append(teacher)
