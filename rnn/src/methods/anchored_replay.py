"""Replay with an anchor on the task-1 RNN state trajectory."""

from typing import Optional

import torch
import torch.nn.functional as F

from src.methods.replay import ReplayMethod
from src.train import compute_loss


class AnchoredReplayMethod(ReplayMethod):
    """Experience replay plus an STPV/hidden-state representation anchor.

    After task 1, a fixed subset of its test trials and their complete hidden
    state trajectories are cached.  Subsequent replay updates add a normalized
    MSE or cosine distance between the current and reference trajectories.
    """

    def __init__(
        self,
        *args,
        anchor_lambda: float = 0.0,
        anchor_loss: str = "mse",
        anchor_probe_size: int = 200,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.anchor_lambda = float(anchor_lambda)
        self.anchor_loss = anchor_loss.lower()
        if self.anchor_loss not in ("mse", "cosine"):
            raise ValueError(
                f"Unknown anchor_loss '{anchor_loss}'. Choose 'mse' or 'cosine'."
            )
        if anchor_probe_size <= 0:
            raise ValueError("anchor_probe_size must be positive.")
        self.anchor_probe_size = int(anchor_probe_size)

        self.probe_inputs: Optional[torch.Tensor] = None
        self.reference_states: Optional[torch.Tensor] = None
        self.reference_state_norm: Optional[torch.Tensor] = None

    @property
    def _anchoring_active(self) -> bool:
        return self.anchor_lambda > 0.0

    def before_task(self, task_idx, task_name, task_gen_fn):
        super().before_task(task_idx, task_name, task_gen_fn)
        if self._anchoring_active and task_idx >= 1:
            print(
                f"  State anchor: lambda={self.anchor_lambda:g}, "
                f"loss={self.anchor_loss}, probe={self.anchor_probe_size}"
            )

    def after_task(self, task_idx, task_name, task_gen_fn):
        """Store replay data and cache task-1 reference states once."""
        super().after_task(task_idx, task_name, task_gen_fn)
        if task_idx == 0 and self._anchoring_active:
            self._cache_reference_states(task_name)

    @torch.no_grad()
    def _cache_reference_states(self, task_name: str) -> None:
        """Cache task-1 test inputs and their deterministic state trajectory."""
        x, _, _ = self.fixed_test_sets[task_name].to_tensor(device=self.device)
        n_probe = min(self.anchor_probe_size, x.shape[1])
        self.probe_inputs = x[:, :n_probe].detach().clone()

        was_training = self.model.training
        self.model.eval()
        try:
            _, states = self.model(self.probe_inputs, return_all_states=True)
            self.reference_states = states.detach().float().clone()
            # Normalize MSE by reference squared norm per trial, averaged over
            # trials. This keeps lambda meaningful across sequence lengths and
            # hidden sizes.
            self.reference_state_norm = (
                self.reference_states.pow(2).sum(dim=(0, 2)).mean().clamp_min(1e-8)
            )
        finally:
            if was_training:
                self.model.train()

        print(
            "  Cached state-anchor reference: "
            f"{n_probe} task-1 trials, trajectory={tuple(self.reference_states.shape)}"
        )

    def _anchor_penalty(self) -> torch.Tensor:
        """Return a differentiable penalty on a random fixed-probe subset."""
        if self.probe_inputs is None or self.reference_states is None:
            raise RuntimeError("Anchor references are not available before task 2.")

        n_probe = self.probe_inputs.shape[1]
        n_batch = min(self.batch_size, n_probe)
        if n_batch < n_probe:
            indices = torch.randperm(n_probe, device=self.probe_inputs.device)[:n_batch]
        else:
            indices = torch.arange(n_probe, device=self.probe_inputs.device)

        probe_inputs = self.probe_inputs.index_select(1, indices)
        reference = self.reference_states.index_select(1, indices)

        # Recurrent process noise is enabled only in train mode. Evaluate mode
        # makes the anchor compare learned representations rather than noise.
        was_training = self.model.training
        self.model.eval()
        try:
            _, current = self.model(probe_inputs, return_all_states=True)
        finally:
            if was_training:
                self.model.train()

        if self.anchor_loss == "mse":
            displacement = (current.float() - reference).pow(2).sum(dim=(0, 2)).mean()
            return displacement / self.reference_state_norm

        current_flat = current.transpose(0, 1).reshape(n_batch, -1)
        reference_flat = reference.transpose(0, 1).reshape(n_batch, -1)
        return (1.0 - F.cosine_similarity(current_flat, reference_flat, dim=1, eps=1e-8)).mean()

    def _replay_train_step(self, optimizer, trial, replay_trials):
        """Replay loss plus a state-trajectory anchor after task 1."""
        self.model.train()
        optimizer.zero_grad()

        x, y, mask = trial.to_tensor(device=self.device)
        outputs = self.model(x, return_all_states=False)
        task_loss = compute_loss(
            outputs, y, mask, loss_type=trial.config.get("loss_type", "cross_entropy")
        )

        for replay_trial in replay_trials:
            rx, ry, rmask = replay_trial.to_tensor(device=self.device)
            replay_outputs = self.model(rx, return_all_states=False)
            task_loss = task_loss + compute_loss(
                replay_outputs,
                ry,
                rmask,
                loss_type=replay_trial.config.get("loss_type", "cross_entropy"),
            )
        task_loss = task_loss / (1 + len(replay_trials))

        if self._anchoring_active and self.probe_inputs is not None:
            anchor_penalty = self._anchor_penalty()
            loss = task_loss + self.anchor_lambda * anchor_penalty
        else:
            loss = task_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()
