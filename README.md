# Representational Drift in Deep Continual Learning

This repo separates incremental training/evaluation from drift analysis. After each incremental task, a checkpoint is saved. Drift analysis then loads checkpoints, extracts layer activations, and computes metrics.

## Setup

```bash
pip install -r requirements.txt
```

## Train (save checkpoints)

```bash
python run_experiment.py --increment 2 --epochs 1 --batch_size 64 --method normal --save_dir experiments/fashion_mnist_incremental
```

Arguments:
- `--increment`: number of new classes per task
- `--method`: `normal` or `replay`
- `--save_dir`: where checkpoints are written

## Analyze Representational Drift

**Note**: This part e is still under debugging and is not recommended for running

```bash
python analyze_drift.py --ckpt_dir experiments/fashion_mnist_incremental --layers network.0,network.2 --max_batches 10 --output drift_results.json
```

Notes:
- Layer names come from `named_modules()` of the model; for the MLP: `network.0` (first Linear), `network.2` (ReLU), etc.
- Metrics reported: mean_shift, cosine_between_means, linear_cka.

## Project Structure

- `src/checkpoints.py`: save/load/list checkpoints with metadata
- `src/continual.py`: incremental training loop (normal/replay) decoupled from saving
- `src/train.py`, `src/eval.py`: training and evaluation utilities
- `src/representations.py`: activation extraction via forward hooks
- `src/drift_metrics.py`: drift metrics (mean shift, cosine, linear CKA)
- `datasets.py`: incremental dataset utilities (FashionMNIST/TinyImageNet)
- `run_experiment.py`: CLI for training and saving checkpoints
- `analyze_drift.py`: CLI for drift analysis across checkpoints
