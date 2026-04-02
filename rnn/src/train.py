import torch
import torch.nn as nn


def compute_loss(outputs, targets, mask, loss_type='cross_entropy'):
    """
    Computes the masked loss for the RNN outputs.

    Args:
        outputs: Tensor of shape (Seq_len, Batch, Output_size)
        targets: Tensor of shape (Seq_len, Batch, Output_size)
        mask: Tensor of shape (Seq_len, Batch) or (Seq_len, Batch, Output_size)
        loss_type: 'cross_entropy' or 'lsq'

    Returns:
        loss: Scalar tensor
    """
    if loss_type == 'lsq':
        raw_loss = nn.functional.mse_loss(outputs, targets, reduction='none')
        masked_loss = raw_loss * mask
        return masked_loss.mean()

    elif loss_type == 'cross_entropy':
        log_preds = torch.log_softmax(outputs, dim=-1)
        raw_loss = -torch.sum(targets * log_preds, dim=-1)
        masked_loss = raw_loss * mask
        return masked_loss.mean()

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def train_step(model, optimizer, trial, device='cpu'):
    """
    Performs a single vanilla training step (no CL penalty).
    CL-specific steps are handled by the respective method classes.

    Args:
        model: CognitiveRNN instance
        optimizer: PyTorch optimizer
        trial: Trial object from datasets.py
        device: 'cpu' or 'cuda'

    Returns:
        loss_val: float
    """
    model.train()
    optimizer.zero_grad()

    x, y, mask = trial.to_tensor(device=device)
    outputs = model(x, return_all_states=False)
    loss = compute_loss(outputs, y, mask, loss_type=trial.config.get('loss_type', 'cross_entropy'))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

