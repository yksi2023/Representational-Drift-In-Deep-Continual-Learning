"""Convert a TF2 SavedModel of BiT (e.g. BiT-S-R50x1) to the .npz format
expected by timm's BiT loader (``timm.models.resnetv2._load_weights``).

The original Google BiT release ships .npz archives whose keys look like::

    resnet/root_block/standardized_conv2d/kernel
    resnet/block1/unit01/a/standardized_conv2d/kernel
    resnet/block1/unit01/a/group_norm/gamma
    ...
    resnet/group_norm/gamma
    resnet/head/conv2d/kernel
    resnet/head/conv2d/bias

TF2 SavedModel variables use the same path names, except each ends with the
TF variable-slot suffix ``:0``. Some TF2 Hub wrappers prepend a Keras layer
prefix like ``keras_layer/...``; this script auto-detects and strips it.

Usage:
    python cnn/tools/convert_bit_tf_to_npz.py \\
        --saved_model pretrained_weights/bit-tensorflow2-s-r50x1-v1 \\
        --out pretrained_weights/BiT-S-R50x1.npz

Only needs tensorflow installed in the current env (``pip install tensorflow-cpu``
is enough; no GPU required for weight conversion).
"""
import argparse
import os
from typing import Dict

import numpy as np

# Silence TF's oneDNN / cpu_feature_guard / absl chatter. Must be set BEFORE
# importing tensorflow for the env-var path to take effect.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


# TF Hub's "feature vector" BiT variants strip the 1000-way classification
# head. timm's loader still requires these keys, so we synthesize zero
# placeholders and rely on the caller to reset_classifier afterwards.
# Head shapes for R50x1 in the original BiT npz layout (HWIO + bias).
_R50x1_HEAD_SHAPES = {
    "resnet/head/conv2d/kernel": (1, 1, 2048, 1000),
    "resnet/head/conv2d/bias":   (1000,),
}


def _strip_slot(name: str) -> str:
    """Drop TF's ':0' variable-slot suffix."""
    return name.split(":")[0] if ":" in name else name


def _detect_prefix(names) -> str:
    """Return a leading path fragment to strip so keys start with ``resnet/``.

    Examples of detected prefixes:
      - "" (already starts with 'resnet/')
      - "keras_layer/"
      - "StatefulPartitionedCall/keras_layer/"
    """
    # First, are any names already in canonical form?
    if any(n.startswith("resnet/") for n in names):
        return ""
    # Otherwise, find the shortest path that, when stripped, leaves a
    # 'resnet/' prefix. We check the first few vars.
    for n in names:
        idx = n.find("resnet/")
        if idx > 0:
            return n[:idx]
    return ""


def convert(saved_model_dir: str, out_path: str, list_only: bool = False) -> None:
    try:
        import tensorflow as tf
        # Also silence absl's Python logger (TF's C++ logs are muted via env).
        tf.get_logger().setLevel("ERROR")
        try:
            from absl import logging as absl_logging
            absl_logging.set_verbosity(absl_logging.ERROR)
        except Exception:
            pass
    except ImportError:
        raise SystemExit(
            "tensorflow is required for this conversion. In your current env:\n"
            "  pip install tensorflow-cpu\n"
            "(CPU-only is sufficient, no GPU needed to extract weights.)"
        )

    print(f"Loading SavedModel from: {saved_model_dir}")
    model = tf.saved_model.load(saved_model_dir)
    variables = list(model.variables)
    if not variables:
        raise SystemExit("No variables found in the SavedModel.")

    raw_names = [_strip_slot(v.name) for v in variables]
    prefix = _detect_prefix(raw_names)
    if prefix:
        print(f"Detected wrapper prefix to strip: '{prefix}'")

    arrays: Dict[str, np.ndarray] = {}
    for v, raw_name in zip(variables, raw_names):
        name = raw_name[len(prefix):] if prefix and raw_name.startswith(prefix) else raw_name
        arr = v.numpy()
        arrays[name] = arr

    print(f"Found {len(arrays)} variables. First 8 entries:")
    for k in list(arrays.keys())[:8]:
        print(f"  {k:<70s}  {arrays[k].shape}  {arrays[k].dtype}")
    if len(arrays) > 8:
        print(f"  ... ({len(arrays) - 8} more)")

    # Feature-vector TF Hub BiT variants drop the 1000-way head. Synthesize
    # zero placeholders so timm's strict loader succeeds; the PyTorch wrapper
    # resets the classifier unconditionally, so these zeros never get used.
    head_added = []
    for key, shape in _R50x1_HEAD_SHAPES.items():
        if key not in arrays:
            arrays[key] = np.zeros(shape, dtype=np.float32)
            head_added.append(key)
    if head_added:
        print(f"Head missing (feature-vector variant); filled {len(head_added)} "
              f"zero placeholder(s) (classifier will be reinit on PyTorch side).")

    if list_only:
        return

    out_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    np.savez(out_abs, **arrays)
    size_mb = os.path.getsize(out_abs) / 1024 ** 2
    print(f"\nWrote {len(arrays)} arrays -> {out_abs} ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saved_model", required=True,
                    help="Extracted SavedModel dir (contains saved_model.pb + variables/).")
    ap.add_argument("--out", required=True,
                    help="Output .npz path (e.g. pretrained_weights/BiT-S-R50x1.npz).")
    ap.add_argument("--list_only", action="store_true",
                    help="Only list variables + shapes; skip writing.")
    args = ap.parse_args()
    convert(args.saved_model, args.out, list_only=args.list_only)


if __name__ == "__main__":
    main()
