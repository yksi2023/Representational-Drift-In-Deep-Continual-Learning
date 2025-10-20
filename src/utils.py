import torch
import numpy as np

def add_heads(model, num_new_classes):
    """Add new output heads to the model for incremental learning."""
    if not hasattr(model, 'network'):
        raise ValueError("Model does not have a 'network' attribute.")
    
    last_layer = model.network[-1]
    if not isinstance(last_layer, torch.nn.Linear):
        raise ValueError("The last layer of the network is not a Linear layer.")
    
    in_features = last_layer.in_features
    out_features = last_layer.out_features
    
    device = last_layer.weight.device

    new_out_features = out_features + num_new_classes
    new_last_layer = torch.nn.Linear(in_features, new_out_features).to(device)
    
    # Copy existing weights and biases
    with torch.no_grad():
        new_last_layer.weight[:out_features] = last_layer.weight
        new_last_layer.bias[:out_features] = last_layer.bias
    
    # Replace the last layer
    model.network[-1] = new_last_layer
    return model


def update_memory(memory_set, new_data, new_labels, memory_size):
    """Update the memory set with new data and labels, maintaining a fixed memory size."""
    combined_data = memory_set["data"] + new_data
    combined_labels = memory_set["labels"] + new_labels
    
    if len(combined_data) > memory_size:
        # Randomly select indices to keep
        indices = np.random.choice(len(combined_data), memory_size, replace=False)
        memory_set["data"] = [combined_data[i] for i in indices]
        memory_set["labels"] = [combined_labels[i] for i in indices]
    else:
        memory_set["data"] = combined_data
        memory_set["labels"] = combined_labels
    
    return memory_set