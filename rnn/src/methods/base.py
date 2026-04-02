import json
import os

import torch
import torch.optim as optim
import numpy as np

from src.train import compute_loss, train_step
from src.eval import generate_fixed_test_set, generate_fixed_train_set, evaluate_model
from src.representations import extract_rnn_representations
from src.checkpoints import save_model


class BaseMethod:
    """
    Base class for continual learning methods on cognitive RNN tasks.
    Defines the main sequential learning loop. Subclasses override hooks
    to implement specific CL strategies (e.g., EWC penalty, replay buffers).
    """

    def __init__(
        self,
        model,
        tasks,
        config,
        num_iterations=2000,
        batch_size=64,
        lr=0.001,
        device='cpu',
        save_dir='./experiments/rnn_drift',
        train_pool_size=200,
        train_seed=12345,
    ):
        self.model = model
        self.tasks = tasks  # List of (task_name, task_generator_fn)
        self.config = config
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.save_dir = save_dir
        self.train_pool_size = train_pool_size

        # Tracking
        self.performance_history = {name: [] for name, _ in tasks}
        self.representations_history = {name: [] for name, _ in tasks}

        # Generate fixed test sets for ALL tasks upfront (seed=42)
        self.fixed_test_sets = {}
        for task_name, task_gen_fn in tasks:
            self.fixed_test_sets[task_name] = generate_fixed_test_set(
                task_gen_fn, config, batch_size=200, seed=42
            )

        # Pre-generate fixed training pools (distinct seed per task, all != 42)
        self.fixed_train_sets = {}
        for i, (task_name, task_gen_fn) in enumerate(tasks):
            task_seed = train_seed + i
            print(f"  Generating fixed training pool for {task_name} (seed={task_seed}, pool={train_pool_size})...")
            self.fixed_train_sets[task_name] = generate_fixed_train_set(
                task_gen_fn, config, batch_size=batch_size,
                pool_size=train_pool_size, seed=task_seed,
            )

        # Pre-cache trial tensors on GPU to avoid repeated CPU→GPU transfers
        if str(device) != 'cpu' and torch.cuda.is_available():
            # Always cache test sets (small: 1 per task)
            print("  Pre-caching test set tensors on GPU...")
            for task_name in self.fixed_test_sets:
                self.fixed_test_sets[task_name].cache_to_device(device)

            # Estimate training pool VRAM: each trial ≈ T*B*(n_in+n_out+1)*4 bytes
            sample_trial = next(iter(self.fixed_train_sets.values()))[0]
            trial_bytes = (sample_trial.x.nbytes + sample_trial.y.nbytes
                           + sample_trial.c_mask.nbytes)
            total_pool = sum(len(v) for v in self.fixed_train_sets.values())
            pool_mb = total_pool * trial_bytes / (1024 ** 2)
            free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)

            if pool_mb < free_mb * 0.6:  # use at most 60% of free VRAM
                print(f"  Pre-caching training pools on GPU ({pool_mb:.0f}MB / {free_mb:.0f}MB free)...")
                for task_name in self.fixed_train_sets:
                    for trial in self.fixed_train_sets[task_name]:
                        trial.cache_to_device(device)
            else:
                print(f"  Skipping GPU cache for training pools ({pool_mb:.0f}MB > 60% of {free_mb:.0f}MB free). "
                      f"Trials will be transferred per-step.")

    def run(self):
        """Main sequential learning loop."""
        self.model.to(self.device)

        for task_idx, (task_name, task_gen_fn) in enumerate(self.tasks):
            print(f"\n{'='*50}")
            print(f"Task {task_idx + 1}/{len(self.tasks)}: {task_name}")
            print(f"{'='*50}")

            # --- Before training hook ---
            self.before_task(task_idx, task_name, task_gen_fn)

            # --- Train on current task (cycle through fixed pool) ---
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            train_pool = self.fixed_train_sets[task_name]
            for i in range(self.num_iterations):
                trial = train_pool[i % self.train_pool_size]
                loss = self.train_step(optimizer, trial)

                if (i + 1) % 200 == 0:
                    print(f"  Iter {i+1}/{self.num_iterations}, Loss: {loss:.4f}")

            # --- After training hook ---
            self.after_task(task_idx, task_name, task_gen_fn)

            # --- Evaluate on ALL tasks and extract representations ---
            self._evaluate_and_record(task_idx, task_name)

            # --- Save checkpoint ---
            save_model(self.model, self.save_dir, task_idx, extra_metadata={'task_name': task_name})

        # --- Persist all results to disk ---
        self._save_results()
        print(f"\nSequential learning complete. Results saved in {self.save_dir}")
        return self.performance_history, self.representations_history

    def _evaluate_and_record(self, task_idx, task_name):
        """Evaluate on ALL tasks, record metrics and representations."""
        print(f"\n--- Evaluation after {task_name} ---")
        for eval_name, _ in self.tasks:
            test_trial = self.fixed_test_sets[eval_name]

            metrics = evaluate_model(self.model, test_trial, device=self.device)
            self.performance_history[eval_name].append(metrics)
            acc_str = f", acc={metrics['accuracy']:.2%}" if not np.isnan(metrics['accuracy']) else ""
            print(f"  {eval_name}: loss={metrics['loss']:.4f}{acc_str}")

            reps = extract_rnn_representations(self.model, test_trial, device=self.device)
            self.representations_history[eval_name].append(reps.cpu().numpy())

    def _save_results(self):
        """Save performance history (JSON) and representations (.npz) to disk."""
        os.makedirs(self.save_dir, exist_ok=True)

        # 1. Performance history → JSON (convert NaN to None for JSON compat)
        def _sanitize(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj

        perf_path = os.path.join(self.save_dir, "performance_history.json")
        with open(perf_path, "w", encoding="utf-8") as f:
            json.dump(_sanitize(self.performance_history), f, indent=2, ensure_ascii=False)
        print(f"Performance history saved to {perf_path}")

        # 2. Representations → one .npz per task
        reps_dir = os.path.join(self.save_dir, "representations")
        os.makedirs(reps_dir, exist_ok=True)
        for task_name, reps_list in self.representations_history.items():
            save_path = os.path.join(reps_dir, f"{task_name}.npz")
            # reps_list[i] is the representation after training on task i
            np.savez_compressed(save_path, **{f"after_task_{i}": r for i, r in enumerate(reps_list)})
        print(f"Representations saved to {reps_dir}/")

    # ------------------------------------------------------------------

    def train_step(self, optimizer, trial):
        """Single training step. Subclasses can override to add penalties."""
        return train_step(self.model, optimizer, trial, device=self.device)

    def before_task(self, task_idx, task_name, task_gen_fn):
        """Hook called before training on a new task. Override in subclasses."""
        pass

    def after_task(self, task_idx, task_name, task_gen_fn):
        """Hook called after training on a task. Override in subclasses."""
        pass
