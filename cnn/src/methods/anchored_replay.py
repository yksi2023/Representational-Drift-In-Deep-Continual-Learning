import torch
from typing import Dict, Any, List, Optional

from src.methods.replay import ReplayMethod
from src.representations import register_activation_hooks


class AnchoredReplayMethod(ReplayMethod):
    """Experience Replay + representation-anchoring penalty (Experiment B).

    Adds the Eq. (23) penalty to the standard replay loss:

        L(theta) = L_replay(theta)
                   + anchor_lambda * mean_{x in P, layer l} d_l(r_theta^l(x), r_1^l(x))

    where P is a fixed probe set drawn from the first task and r_1 are the
    activations captured at checkpoint theta_1 (after task 0). The penalty acts
    on the *representation*, not on the weights, so the network stays plastic --
    this is what decouples drift from weight plasticity and distinguishes the
    manipulation from EWC.

    Setting ``anchor_lambda = 0`` recovers plain replay exactly (no probe
    caching, no extra forward), so the lambda=0 arm of Experiment B can use
    either ``replay`` or ``anchored_replay --anchor_lambda 0`` interchangeably.

    The per-layer distance ``d_l`` is switchable via ``anchor_loss``:
      * ``mse`` (default, paper Eq. 23): squared L2, normalized per layer by the
        mean ||r_1||^2 so shallow high-dimensional stages don't dominate.
      * ``cosine``: 1 - cos(r_theta, r_1), mirroring the primary cosine drift
        metric and naturally scale/dimension invariant.
    """

    def __init__(
        self,
        *args,
        anchor_lambda: float = 0.0,
        anchor_layers: str = "layer3,layer4",
        anchor_loss: str = "mse",
        anchor_probe_size: int = 256,
        anchor_probe_mode: str = "test",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.anchor_lambda = float(anchor_lambda)
        self.anchor_layers: List[str] = [s.strip() for s in anchor_layers.split(",") if s.strip()]
        self.anchor_loss = anchor_loss.lower()
        if self.anchor_loss not in ("mse", "cosine"):
            raise ValueError(f"Unknown anchor_loss '{anchor_loss}'. Choose 'mse' or 'cosine'.")
        self.anchor_probe_size = int(anchor_probe_size)
        self.anchor_probe_mode = anchor_probe_mode

        # Populated after task 0 (checkpoint theta_1) when anchoring is active.
        self.probe_inputs: Optional[torch.Tensor] = None
        self.r1_acts: Dict[str, torch.Tensor] = {}
        self.r1_norms: Dict[str, float] = {}

    @property
    def _anchoring_active(self) -> bool:
        return self.anchor_lambda > 0.0 and len(self.anchor_layers) > 0

    def get_training_params(self) -> Dict[str, Any]:
        params = super().get_training_params()
        params["anchor_lambda"] = self.anchor_lambda
        params["anchor_layers"] = ",".join(self.anchor_layers)
        params["anchor_loss"] = self.anchor_loss
        params["anchor_probe_size"] = self.anchor_probe_size
        params["anchor_probe_mode"] = self.anchor_probe_mode
        return params

    def _get_extra_metadata(self) -> Optional[Dict]:
        return {
            "anchor_lambda": self.anchor_lambda,
            "anchor_layers": ",".join(self.anchor_layers),
            "anchor_loss": self.anchor_loss,
        }

    def _print_task_info(self, task_idx: int) -> None:
        super()._print_task_info(task_idx)
        if self._anchoring_active:
            print(f"Anchor: lambda={self.anchor_lambda}, loss={self.anchor_loss}, "
                  f"layers={self.anchor_layers}, probe={self.anchor_probe_size}")

    def after_task(self, task_idx: int, train_loader) -> None:
        """Update replay memory, and after task 0 cache the probe set + r_1."""
        super().after_task(task_idx, train_loader)
        if task_idx == 0 and self._anchoring_active:
            self._cache_probe_references()

    @torch.no_grad()
    def _cache_probe_references(self) -> None:
        """Build a fixed probe set from task-1 data and store r_1 activations."""
        probe_loader = self.experiment_dataset.get_loader(
            mode=self.anchor_probe_mode,
            label=range(self.increment),
            batch_size=self.batch_size,
            shuffle=False,
        )

        inputs_buf: List[torch.Tensor] = []
        collected = 0
        for inputs, _labels in probe_loader:
            inputs_buf.append(inputs)
            collected += inputs.shape[0]
            if collected >= self.anchor_probe_size:
                break
        probe_inputs = torch.cat(inputs_buf, dim=0)[: self.anchor_probe_size]
        self.probe_inputs = probe_inputs.to(self.device, non_blocking=True)

        was_training = self.model.training
        self.model.eval()
        activations, handles = register_activation_hooks(
            self.model, self.anchor_layers, detach=True
        )
        try:
            _ = self.model(self.probe_inputs)
            for name in self.anchor_layers:
                act = activations[name].detach().float()
                self.r1_acts[name] = act
                # Per-layer normalizer: mean squared norm of the reference
                # activations. Keeps the MSE term O(1) and comparable across
                # layers of very different dimensionality.
                self.r1_norms[name] = float(act.pow(2).sum(dim=1).mean().clamp_min(1e-8))
        finally:
            for h in handles:
                h.remove()
        if was_training:
            self.model.train()

        print(f"Cached anchor references: {self.probe_inputs.shape[0]} probe samples, "
              f"layers={self.anchor_layers}")

    def _anchor_penalty(self) -> torch.Tensor:
        """Differentiable representation-anchoring penalty over the probe set."""
        # Stochastic estimate: forward a random probe mini-batch each step.
        n = self.probe_inputs.shape[0]
        k = min(self.batch_size, n)
        if k < n:
            idx = torch.randperm(n, device=self.device)[:k]
        else:
            idx = torch.arange(n, device=self.device)
        probe_batch = self.probe_inputs.index_select(0, idx)

        activations, handles = register_activation_hooks(
            self.model, self.anchor_layers, detach=False
        )
        try:
            _ = self.model(probe_batch)
            per_layer = []
            for name in self.anchor_layers:
                cur = activations[name].float()
                ref = self.r1_acts[name].index_select(0, idx)
                if self.anchor_loss == "mse":
                    # Mean squared L2 displacement, normalized per layer.
                    d = (cur - ref).pow(2).sum(dim=1).mean() / self.r1_norms[name]
                else:  # cosine
                    cos = torch.nn.functional.cosine_similarity(cur, ref, dim=1, eps=1e-8)
                    d = (1.0 - cos).mean()
                per_layer.append(d)
            penalty = torch.stack(per_layer).mean()
        finally:
            for h in handles:
                h.remove()
        return penalty

    def compute_loss(self, outputs, labels, active_range, task_idx, inputs):
        """Replay task loss + anchor_lambda * representation-anchoring penalty."""
        task_loss = self._task_loss(outputs, labels, active_range)

        use_anchor = (
            self._anchoring_active and task_idx >= 1 and self.probe_inputs is not None
        )
        if use_anchor:
            penalty = self._anchor_penalty()
        else:
            penalty = outputs.new_tensor(0.0)

        loss = task_loss + self.anchor_lambda * penalty
        return loss, {"Task Loss": task_loss.item(), "Anchor Loss": float(penalty.item())}
