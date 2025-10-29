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
    if not new_data:
        return memory_set
    
    combined_data = memory_set["data"] + new_data
    combined_labels = memory_set["labels"] + new_labels

    # Class-balanced retention: allocate roughly equal quota per seen class
    from collections import defaultdict
    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(combined_labels):
        label_to_indices[int(lbl)].append(idx)

    seen_labels = sorted(label_to_indices.keys())
    if len(seen_labels) == 0:
        return {"data": [], "labels": []}

    per_class_quota = max(1, memory_size // len(seen_labels))

    selected_indices = []
    # First pass: take up to per_class_quota from each class
    for lbl in seen_labels:
        indices = label_to_indices[lbl]
        if len(indices) <= per_class_quota:
            selected_indices.extend(indices)
        else:
            chosen = np.random.choice(indices, per_class_quota, replace=False)
            selected_indices.extend(chosen.tolist())

    # If we still have room (because some classes had < quota), fill with leftovers
    if len(selected_indices) < memory_size:
        already = set(selected_indices)
        leftovers = [idx for lbl in seen_labels for idx in label_to_indices[lbl] if idx not in already]
        need = min(memory_size - len(selected_indices), len(leftovers))
        if need > 0:
            extra = np.random.choice(leftovers, need, replace=False)
            selected_indices.extend(extra.tolist())

    # If we exceeded memory_size due to rounding, trim uniformly at random
    if len(selected_indices) > memory_size:
        selected_indices = np.random.choice(selected_indices, memory_size, replace=False).tolist()

    updated_memory = {
        "data": [combined_data[i] for i in selected_indices],
        "labels": [combined_labels[i] for i in selected_indices]
    }

    return updated_memory