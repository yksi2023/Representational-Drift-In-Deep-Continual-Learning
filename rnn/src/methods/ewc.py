import numpy as np
import torch
from src.methods.base import BaseMethod
from src.train import compute_loss


def _sample_observation_loss(outputs, targets, mask, loss_type, rng):
    """Sample one scored timestep and return its per-observation loss."""
    if mask.ndim == 2:
        time_weights = mask[:, 0]
    elif mask.ndim == 3:
        time_weights = mask[:, 0].mean(dim=-1)
    else:
        raise ValueError(f"Expected a 2D or 3D mask, got shape {tuple(mask.shape)}")

    weights = time_weights.detach().float().cpu().numpy().astype(np.float64)
    weights = np.clip(weights, 0.0, None)
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("Cannot estimate Fisher from a trial with an empty loss mask")

    time_idx = int(rng.choice(len(weights), p=weights / total_weight))

    if loss_type == 'cross_entropy':
        log_probs = torch.log_softmax(outputs[time_idx, 0], dim=-1)
        return -(targets[time_idx, 0] * log_probs).sum()

    if loss_type == 'lsq':
        squared_error = (outputs[time_idx, 0] - targets[time_idx, 0]).pow(2)
        if mask.ndim == 3:
            output_weights = mask[time_idx, 0]
            return (squared_error * output_weights).sum() / output_weights.sum().clamp_min(1e-8)
        return squared_error.mean()

    raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_fisher_information(
    model,
    task_generator,
    config,
    num_samples,
    device='cpu',
    fisher_seed=0,
):
    """
    Estimate diagonal empirical Fisher from individual scored timesteps.

    Each Fisher sample uses one trial and one timestep sampled in proportion to
    the training loss mask. Squaring this per-observation gradient avoids
    cancellation between different timesteps in a sequence-averaged loss.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    was_training = model.training
    model.eval()
    fisher_dict = {}
    optpar_dict = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            optpar_dict[name] = param.detach().clone()
            fisher_dict[name] = torch.zeros_like(param)

    fisher_config = config.copy()
    fisher_config['rng'] = np.random.RandomState(fisher_seed)
    time_rng = np.random.RandomState(fisher_seed + 1)

    try:
        for _ in range(num_samples):
            model.zero_grad(set_to_none=True)
            trial = task_generator(fisher_config, batch_size=1, mode='random')
            x, y, mask = trial.to_tensor(device=device)
            outputs = model(x)
            loss = _sample_observation_loss(
                outputs,
                y,
                mask,
                loss_type=trial.config.get('loss_type', 'cross_entropy'),
                rng=time_rng,
            )
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name].add_(param.grad.detach().pow(2))
    finally:
        model.train(was_training)

    for fisher in fisher_dict.values():
        fisher.div_(num_samples)

    return fisher_dict, optpar_dict


def _fisher_stats(fisher_dict):
    total = sum(value.sum().item() for value in fisher_dict.values())
    count = sum(value.numel() for value in fisher_dict.values())
    maximum = max((value.max().item() for value in fisher_dict.values()), default=0.0)
    nonzero = sum(torch.count_nonzero(value).item() for value in fisher_dict.values())
    return total / max(count, 1), maximum, nonzero / max(count, 1)


def ewc_penalty(model, fisher_dict, optpar_dict, ewc_lambda):
    """Computes the EWC penalty: (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2."""
    penalty = 0.0
    for name, param in model.named_parameters():
        if name in fisher_dict and param.requires_grad:
            penalty += (fisher_dict[name] * (param - optpar_dict[name]).pow(2)).sum()
    return ewc_lambda * 0.5 * penalty


class EWCMethod(BaseMethod):
    """Elastic Weight Consolidation for sequential cognitive task learning."""

    def __init__(self, ewc_lambda=100.0, fisher_samples=200, **kwargs):
        super().__init__(**kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.ewc_tasks = []  # list of (fisher_dict, optpar_dict), one per completed task

    def train_step(self, optimizer, trial):
        """Training step with EWC penalty added to the task loss."""
        self.model.train()
        optimizer.zero_grad()

        x, y, mask = trial.to_tensor(device=self.device)
        outputs = self.model(x, return_all_states=False)
        loss = compute_loss(outputs, y, mask, loss_type=trial.config.get('loss_type', 'cross_entropy'))

        for fisher_dict, optpar_dict in self.ewc_tasks:
            loss += ewc_penalty(self.model, fisher_dict, optpar_dict, self.ewc_lambda)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    def after_task(self, task_idx, task_name, task_gen_fn):
        """Compute Fisher Information after each task and store separately."""
        print(f"  Computing Fisher Information for {task_name} (total EWC terms: {task_idx + 1})...")
        fisher, optpar = compute_fisher_information(
            self.model,
            task_gen_fn,
            self.config,
            self.fisher_samples,
            self.device,
            fisher_seed=100_000 + task_idx,
        )
        fisher_mean, fisher_max, fisher_nonzero = _fisher_stats(fisher)
        print(f"    Fisher mean={fisher_mean:.3e}, max={fisher_max:.3e}, "
              f"nonzero={fisher_nonzero:.1%}")
        self.ewc_tasks.append((fisher, optpar))
