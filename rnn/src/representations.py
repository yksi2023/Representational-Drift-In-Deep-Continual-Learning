import torch

def extract_rnn_representations(model, trial, device='cpu'):
    """
    Extract the Spatiotemporal Population Vector (STPV) from the RNN.
    The STPV is formed by concatenating Population Vectors (hidden states)
    across all time steps into a single long vector per trial.
    
    Args:
        model: Trained CognitiveRNN model
        trial: Trial object (contains input data)
        device: 'cpu' or 'cuda'
        
    Returns:
        stpv: Tensor of shape (Batch, Seq_len * Hidden_size).
    """
    model.eval()
    
    # Get inputs and pass through the model to get all hidden states
    x, _, _ = trial.to_tensor(device=device)
    
    with torch.no_grad():
        # outputs_tensor shape: (Seq_len, Batch, Output_size)
        # states_tensor shape: (Seq_len, Batch, Hidden_size)
        _, states_tensor = model(x, return_all_states=True)
        
    # Convert from (Seq_len, Batch, Hidden_size) to (Batch, Seq_len, Hidden_size)
    states_tensor = states_tensor.transpose(0, 1)
    
    # Flatten the sequence and hidden dimensions into a single long vector per batch item
    batch_size = states_tensor.size(0)
    representations = states_tensor.reshape(batch_size, -1)
    
    return representations
