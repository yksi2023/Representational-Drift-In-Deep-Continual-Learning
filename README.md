# Representational Drift in Deep Continual Learning

Tools for analyzing representational drift in deep continual learning, separating incremental training from post-hoc analysis.

## Usage

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Train (Generate Checkpoints)
```bash
python run_experiment.py --increment 2 --method normal --save_dir experiments/fashion_mnist
```

### 3. Analyze Drift
*Note: This module is currently experimental.*
```bash
python analyze_drift.py --ckpt_dir experiments/fashion_mnist --layers network.0,network.2 --output drift_results.json
```

## Structure
- **CLI**: `run_experiment.py` (train), `analyze_drift.py` (analyze)
- **Core**: `src/continual.py` (training), `src/drift_metrics.py` (metrics), `src/checkpoints.py` (persistence)
- **Data**: `datasets.py` (incremental loaders)
