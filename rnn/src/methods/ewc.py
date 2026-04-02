import torch
from src.methods.base import BaseMethod
from src.train import compute_loss, train_step


def compute_fisher_information(model, task_generator, config, num_samples, device='cpu'):
    """
    Computes the diagonal Fisher Information matrix for the RNN parameters.
    Used by the EWC method to estimate parameter importance.
    """
    model.eval()
    fisher_dict = {}
    optpar_dict = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            optpar_dict[name] = param.data.clone().detach()
            fisher_dict[name] = torch.zeros_like(param.data)

    for _ in range(num_samples):
        model.zero_grad()
        trial = task_generator(config, batch_size=1, mode='random')
        x, y, mask = trial.to_tensor(device=device)
        outputs = model(x)
        loss = compute_loss(outputs, y, mask, loss_type=config.get('loss_type', 'cross_entropy'))
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2) / num_samples

    return fisher_dict, optpar_dict


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
        self.fisher_dict = None
        self.optpar_dict = None

    def train_step(self, optimizer, trial):
        """Training step with EWC penalty added to the task loss."""
        self.model.train()
        optimizer.zero_grad()

        x, y, mask = trial.to_tensor(device=self.device)
        outputs = self.model(x, return_all_states=False)
        loss = compute_loss(outputs, y, mask, loss_type=trial.config.get('loss_type', 'cross_entropy'))

        if self.fisher_dict is not None and self.optpar_dict is not None:
            loss += ewc_penalty(self.model, self.fisher_dict, self.optpar_dict, self.ewc_lambda)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    def after_task(self, task_idx, task_name, task_gen_fn):
        """Compute and accumulate Fisher Information after each task."""
        print(f"  Computing Fisher Information for {task_name}...")
        new_fisher, new_optpar = compute_fisher_information(
            self.model, task_gen_fn, self.config, self.fisher_samples, self.device
        )

        if self.fisher_dict is None:
            self.fisher_dict = new_fisher
            self.optpar_dict = new_optpar
        else:
            for name in self.fisher_dict:
                self.fisher_dict[name] += new_fisher[name]
                self.optpar_dict[name] = new_optpar[name]
