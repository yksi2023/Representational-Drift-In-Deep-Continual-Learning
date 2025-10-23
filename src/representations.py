from typing import Dict, List, Tuple, Callable, Optional

import torch


def register_activation_hooks(
    model: torch.nn.Module,
    layer_names: List[str],
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """Register forward hooks to capture activations for the given layer names.

    Returns a dict to store activations and the list of hook handles.
    """
    activations: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    name_to_module: Dict[str, torch.nn.Module] = dict(model.named_modules())
    for name in layer_names:
        if name not in name_to_module:
            raise ValueError(f"Layer name '{name}' not found in model modules.")

        def _make_hook(key: str) -> Callable:
            def _hook(_module, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                activations[key] = out.detach().flatten(1).cpu()
            return _hook

        handle = name_to_module[name].register_forward_hook(_make_hook(name))
        handles.append(handle)

    return activations, handles


@torch.no_grad()
def extract_representations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_names: List[str],
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Run data through the model and collect activations from specified layers.

    Returns a dict mapping layer name to tensor of shape [N, D].
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    collected: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
    activations, handles = register_activation_hooks(model, layer_names)

    try:
        for batch_idx, (inputs, _labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            _ = model(inputs)
            for ln in layer_names:
                if ln in activations:
                    collected[ln].append(activations[ln])
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    finally:
        for h in handles:
            h.remove()

    return {ln: torch.cat(tensors, dim=0) if tensors else torch.empty(0) for ln, tensors in collected.items()}