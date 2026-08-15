# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Environment

```bash
conda activate drift
pip install -r requirements.txt   # torch 2.9, torchvision, tqdm, matplotlib, numpy, scikit-learn, scipy, timm
```

## Commands

### CNN Experiments

```bash
# Train all CL methods on TinyImageNet (pretrained ResNet-18)
bash cnn.sh <i>              # e.g. bash cnn.sh 1 → cnn/experiments/exp1_cnn_{normal,replay,ewc,lwf,gpm}

# Analyze drift for all exp<i>_cnn_* directories (auto-detects probe layers)
bash analysis_cnn.sh <i>     # e.g. bash analysis_cnn.sh 1

# Train BiT-S R50x1 on ImageNet-21k-P200
bash run_imagenet21k_p200.sh <i>

# Single CNN run (inside cnn/):
python run_experiment.py --method replay --dataset tiny_imagenet --model resnet18_pretrained \
    --increment 10 --epochs 50 --freeze_until layer2 --channels_last --amp --scheduler cosine \
    --save_dir experiments/my_exp

# Single drift analysis (inside cnn/):
python analyze_drift.py --ckpt_dir experiments/my_exp --layers "backbone.layer3.0.relu,backbone.layer4.0.relu"
```

### RNN Experiments

```bash
bash rnn.sh <i>              # 5 methods (normal, replay, ewc, lwf, hypernet) on 18 cognitive tasks
bash analysis_rnn.sh <i>     # Drift analysis for all exp<i>_rnn_* directories

# Single RNN run (inside rnn/):
python run_experiment.py --method replay --memory_per_task 50 --save_dir experiments/my_rnn_exp
```

## Architecture

The project studies how internal representations drift during continual learning. It has two parallel experiment stacks (CNN and RNN) that share the same conceptual design but differ in implementation details.

### Shared pattern (both `cnn/` and `rnn/`)

1. **`run_experiment.py`** — CLI entry point. Parses args, builds model + dataset, delegates to `src/continual.py`.
2. **`src/continual.py`** — Thin orchestrator. Routes to the right method class via `src/methods/__init__.py`'s `get_method()`.
3. **`src/methods/base.py`** — `BaseContinualMethod` (ABC). Owns the main `run()` loop: for each task → `before_task()` → `train_task()` → `after_task()` → `save_checkpoint()` → `_evaluate_and_record_all()`.
4. **`src/methods/{normal,replay,ewc,lwf,gpm,hypernet}.py`** — Concrete method implementations, overriding `train_task()` and hooks.
5. **`analyze_drift.py`** — Reads checkpoints, builds representation cache, runs 6-stage drift analysis.
6. **`src/analysis/cache.py`** — `build_reps_cache()`: one forward pass per checkpoint, extracts activations via hooks.
7. **`src/representations.py`** — `register_activation_hooks()` + `extract_representations()`. Hooks capture activations from named modules; tensors stay on GPU inside the forward pass then move to CPU per-batch.
8. **`src/drift_metrics.py`** — Low-level: per-sample cosine similarity and L2 distance between two tensors of shape `[N, D]`.

### CNN-specific

- **`src/models.py`** — 5 models via `build_model()` registry: MLP (`FashionMNISTModel`), `ResNet18_Tiny`, `PretrainedResNet18` (torchvision ResNet-18), `ResNet18CIFAR_GN` (StdConv2d + GroupNorm + Zero-gamma init), `BiTResNet50_IN1k` (timm BiT-S R50x1 with GN+WS, pretrained weights from Google's `.npz`).
- **`datasets.py`** — `IncrementalFashionMNIST`, `IncrementalTinyImageNet`, `IncrementalCIFAR100`, `IncrementalImageNet21kP200`. All follow the same interface: `get_set(mode, label)` + `get_loader(mode, label, batch_size)` with precomputed class indices for fast per-task subsetting.
- **`MODEL_DEFAULTS`** dict maps dataset → `{model, num_classes, img_size}`. Used by both `run_experiment.py` (training) and `analyze_drift.py` (reconstructing model from config).
- Checkpoints saved as `model_after_task_N.pth` + companion `model_after_task_N.json` metadata.
- TF32 enabled via `torch.backends.cudnn.conv.fp32_precision = 'tf32'` and `torch.backends.cuda.matmul.fp32_precision = 'tf32'` (the new PyTorch 2.9 API — do NOT use the legacy `allow_tf32` flags).

### RNN-specific

- **`src/models.py`** — `CognitiveRNN` (continuous-time RNN with configurable activation, recurrence initialization, time constants).
- **`datasets.py`** — Generates synthetic cognitive tasks (perceptual decision-making, context integration, etc.) from Yang et al. 2019. Tasks are pre-generated pools of trials.
- **`src/hypernet.py`** — HyperNetwork implementation (task-conditioned weight generation), unique to RNN.
- RNN drift analysis adds `temporal_correlation.py` and `vector_drift.py` which CNN doesn't have.

### Experiment bookkeeping

- Each experiment writes `experiment_config.json` (all CLI flags) so analysis can reconstruct dataset/model/classes.
- Training produces `training_metrics.json`, `performance_history.json`, and `comprehensive_evaluation.json`.
- Drift analysis outputs go to `<ckpt_dir>/drift_analysis/` by default.

### Model freezing

`freeze_until` takes a layer name (model-specific) and freezes all layers up to and including it. Frozen BN/GN layers are kept in eval mode during training. Layer names follow each model's `_layer_order` list.
