# Representational Drift in Deep Continual Learning

Train continual learning models (CNN / RNN) and analyze how their internal representations drift across tasks.

## Setup

```bash
pip install -r requirements.txt
conda activate drift
```

## Quick Start

### CNN — Train + Analyze

```bash
# Train all methods on TinyImageNet (ResNet-18 pretrained)
bash cnn.sh 1

# Analyze drift (auto-detects probe layers per model)
bash analysis_cnn.sh 1
```

### RNN — Train + Analyze

```bash
# Train all methods on Yang et al. sequential tasks
bash rnn.sh 1

# Analyze drift
bash analysis_rnn.sh 1
```

### ImageNet-21k-P200 (BiT-S R50x1)

```bash
bash run_imagenet21k_p200.sh 1
bash analysis_cnn.sh 1
```

## Supported Methods

| Method | CNN | RNN |
|--------|-----|-----|
| Normal (fine-tuning) | ✓ | ✓ |
| Replay | ✓ | ✓ |
| EWC | ✓ | ✓ |
| LwF | ✓ | ✓ |
| GPM | ✓ | — |
| HyperNet | — | ✓ |

## Supported Models & Datasets

**CNN**
- Models: MLP, ResNet-18 (from scratch / pretrained / CIFAR-GN), BiT-S R50x1
- Datasets: FashionMNIST, TinyImageNet, CIFAR-100, ImageNet-21k-P200

**RNN**
- Model: Cognitive RNN (Yang et al. 2019)
- Tasks: 18 default cognitive tasks (perceptual decision-making, context integration, etc.)

## Drift Analysis Pipeline

`analyze_drift.py` runs 6 analysis stages per checkpoint:

1. **Baseline drift** — cosine / L2 distance vs first checkpoint
2. **Model similarity** — pairwise cosine similarity matrix + decay profile
3. **Sample similarity** — per-sample cosine similarity matrices
4. **Subspace drift** — coding/null subspace decomposition via PCA
5. **Gap drift** — Sample-PV & ERV correlation vs task gap
6. **Performance** — accuracy curves and task × stage heatmap

## Project Structure

```
cnn/                          # CNN experiments
  run_experiment.py           # Training entry point
  analyze_drift.py            # Analysis entry point
  datasets.py                 # Data loaders
  src/
    models.py                 # Model definitions
    continual.py              # Method dispatch
    methods/                  # CL method implementations
    analysis/                 # Drift analysis modules
    representations.py        # Activation hooks
    checkpoints.py            # Checkpoint I/O

rnn/                          # RNN experiments (same layout)
  run_experiment.py
  analyze_drift.py
  ...
```

## Key CLI Flags (CNN)

```
--method {normal,replay,ewc,lwf,gpm}
--dataset {fashion_mnist,tiny_imagenet,cifar100,imagenet21k_p200}
--model {mlp,resnet18_tiny,resnet18_pretrained,resnet18_cifar_gn,bit_s_r50x1_in1k}
--increment N          # classes per task
--learning_mode {til,cil}
--freeze_until LAYER   # freeze backbone up to layer
--amp                  # mixed precision (not recommended for analysis)
```
