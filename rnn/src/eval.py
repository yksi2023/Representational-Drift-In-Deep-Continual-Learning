import torch
import numpy as np
from src.train import compute_loss

def generate_fixed_test_set(generate_task_fn, config, batch_size, seed=42):
    """
    Generates a fixed test set for a specific task to ensure consistent evaluation
    and representation extraction.
    
    Args:
        generate_task_fn: Function to generate trials
        config: Task configuration
        batch_size: Size of the test set
        seed: Random seed for reproducibility
        
    Returns:
        trial: A fixed Trial object
    """
    # Create a new config with a fixed random seed
    fixed_config = config.copy()
    fixed_config['rng'] = np.random.RandomState(seed)
    
    # Generate the trial
    trial = generate_task_fn(fixed_config, batch_size, mode='random')
    return trial


def generate_fixed_train_set(generate_task_fn, config, batch_size, pool_size, seed=12345):
    """
    Pre-generates a fixed pool of training batches for a task.

    Args:
        generate_task_fn: Function to generate trials
        config: Task configuration
        batch_size: Batch size per trial
        pool_size: Number of batches to pre-generate
        seed: Random seed (must differ from test set seed=42)

    Returns:
        trials: List of Trial objects (length = pool_size)
    """
    fixed_config = config.copy()
    fixed_config['rng'] = np.random.RandomState(seed)

    trials = []
    for _ in range(pool_size):
        trial = generate_task_fn(fixed_config, batch_size, mode='random')
        trials.append(trial)
    return trials

def popvec_decode(output_ring, pref):
    """Decode angle from population vector (ring output units).

    Args:
        output_ring: (..., n_eachring) array or tensor of ring unit activations.
        pref: (n_eachring,) array of preferred directions in [0, 2π).

    Returns:
        angles: (...,) decoded angles in [0, 2π).
    """
    if isinstance(output_ring, torch.Tensor):
        output_ring = output_ring.cpu().numpy()
    pref = np.array(pref)
    # Weighted circular mean
    cos_sum = (output_ring * np.cos(pref)).sum(axis=-1)
    sin_sum = (output_ring * np.sin(pref)).sum(axis=-1)
    angles = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)
    return angles


def evaluate_model(model, trial, device='cpu'):
    """
    Evaluates the model on a given trial (usually a fixed test set).
    
    Args:
        model: Trained CognitiveRNN
        trial: Trial object
        device: 'cpu' or 'cuda'
        
    Returns:
        metrics: Dictionary containing loss, accuracy, fixation accuracy
    """
    model.eval()
    
    x, y, mask = trial.to_tensor(device=device)
    
    with torch.no_grad():
        outputs = model(x, return_all_states=False)
        loss = compute_loss(outputs, y, mask, loss_type=trial.config.get('loss_type', 'cross_entropy'))

    # PopVec accuracy: decode output ring and compare to target angle
    # Apply softmax first (model outputs raw logits for cross_entropy)
    output_probs = torch.softmax(outputs, dim=-1).cpu().numpy()  # (T, B, n_output)
    y_loc = trial.y_loc                        # (T, B), target angle or -1 for fixation
    pref = trial.pref                          # (n_eachring,)
    n_eachring = trial.n_eachring

    # Response timesteps: y_loc >= 0
    resp_mask = y_loc >= 0
    # Fixation timesteps: y_loc < 0
    fix_mask = y_loc < 0

    # Angular accuracy on response timesteps
    accuracy = np.nan
    if resp_mask.any():
        resp_outputs_ring = output_probs[resp_mask][:, 1:]  # strip fixation unit
        decoded = popvec_decode(resp_outputs_ring, pref)
        target = y_loc[resp_mask]
        # Angular distance
        ang_dist = np.abs(decoded - target)
        ang_dist = np.minimum(ang_dist, 2 * np.pi - ang_dist)
        accuracy = float(np.mean(ang_dist < (2 * np.pi / n_eachring)))

    # Fixation accuracy: fixation unit should have highest probability
    fix_accuracy = np.nan
    if fix_mask.any():
        fix_outputs = output_probs[fix_mask]  # (N, n_output)
        fix_unit = fix_outputs[:, 0]
        ring_max = fix_outputs[:, 1:].max(axis=-1)
        fix_accuracy = float(np.mean(fix_unit > ring_max))

    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy,
        'fix_accuracy': fix_accuracy,
    }
    
    return metrics
