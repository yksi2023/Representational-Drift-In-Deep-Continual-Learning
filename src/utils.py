import torch


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