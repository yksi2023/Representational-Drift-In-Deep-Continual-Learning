"""
Hypernetwork-based continual learning (von Oswald et al., 2020).

A chunked hypernetwork generates the target CognitiveRNN's weights from a
per-task embedding.  Forgetting is prevented by an output-regularization loss
that penalises changes to the hypernetwork's output for previously learned
task embeddings.
"""

import torch
import torch.optim as optim
import numpy as np

from src.methods.base import BaseMethod
from src.train import compute_loss
from src.hypernet import ChunkedHyperNetwork, get_target_param_info
from src.eval import evaluate_model
from src.representations import extract_rnn_representations
from src.checkpoints import save_model


class HyperNetMethod(BaseMethod):
    """Hypernetwork continual learning for cognitive RNNs."""

    def __init__(
        self,
        hnet_emb_dim=64,
        hnet_chunk_emb_dim=64,
        hnet_hidden=128,
        hnet_chunks=10,
        hnet_beta=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hnet_beta = hnet_beta
        self.emb_dim = hnet_emb_dim

        # Target model parameter layout
        self.target_num_params, self.param_info = get_target_param_info(self.model)
        print(f"  HyperNet target params: {self.target_num_params}")

        # Build chunked hypernetwork
        self.hnet = ChunkedHyperNetwork(
            target_num_params=self.target_num_params,
            num_chunks=hnet_chunks,
            emb_dim=hnet_emb_dim,
            chunk_emb_dim=hnet_chunk_emb_dim,
            hyper_hidden=hnet_hidden,
        ).to(self.device)

        # Task embeddings (created on demand)
        self.task_embs = []         # list of nn.Parameter, one per task
        # Stored weight snapshots for output regularisation
        self.stored_targets = []    # list of detached tensors, one per completed task

    # ------------------------------------------------------------------
    # Weight generation helpers
    # ------------------------------------------------------------------

    def _create_task_emb(self):
        """Create a new learnable task embedding."""
        emb = torch.randn(self.emb_dim, device=self.device) * 0.01
        emb = torch.nn.Parameter(emb)
        self.task_embs.append(emb)
        return emb

    def _load_generated_weights(self, task_emb):
        """Generate weights from hnet and load into self.model (in-place)."""
        self.hnet.generate_and_load(task_emb, self.model, self.param_info)

    def _output_reg_loss(self, current_task_idx):
        """Compute output regularisation: MSE between hnet output for old
        embeddings and their stored snapshots."""
        if not self.stored_targets:
            return torch.tensor(0.0, device=self.device)
        reg = torch.tensor(0.0, device=self.device)
        for i in range(current_task_idx):
            predicted = self.hnet(self.task_embs[i])
            target = self.stored_targets[i]
            reg += torch.nn.functional.mse_loss(predicted, target)
        return reg / current_task_idx

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run(self):
        """Custom run loop: optimise hypernetwork + task embeddings."""
        self.model.to(self.device)

        for task_idx, (task_name, task_gen_fn) in enumerate(self.tasks):
            print(f"\n{'='*50}")
            print(f"Task {task_idx + 1}/{len(self.tasks)}: {task_name}")
            print(f"{'='*50}")

            # Create embedding for new task
            task_emb = self._create_task_emb()

            # Optimiser: hnet params + all task embeddings (including new one)
            opt_params = list(self.hnet.parameters()) + self.task_embs
            optimizer = optim.Adam(opt_params, lr=self.lr)

            self.before_task(task_idx, task_name, task_gen_fn)

            train_pool = self.fixed_train_sets[task_name]
            for i in range(self.num_iterations):
                loss_val = self._hnet_train_step(
                    optimizer, train_pool[i % self.train_pool_size], task_idx
                )
                if (i + 1) % 200 == 0:
                    print(f"  Iter {i+1}/{self.num_iterations}, Loss: {loss_val:.4f}")

            # Snapshot hnet output for this task (for future regularisation)
            with torch.no_grad():
                snapshot = self.hnet(task_emb).detach().clone()
            self.stored_targets.append(snapshot)

            self.after_task(task_idx, task_name, task_gen_fn)

            # Evaluate: for each eval task, load weights from its own embedding
            # (or current embedding if not yet learned)
            self._evaluate_hypernet(task_idx, task_name)

            # Save target model checkpoint (with current task's generated weights)
            self._load_generated_weights(task_emb)
            save_model(self.model, self.save_dir, task_idx,
                       extra_metadata={'task_name': task_name})

            # Also save hypernetwork + embeddings
            self._save_hnet_checkpoint(task_idx)

        self._save_results()
        print(f"\nSequential learning complete. Results saved in {self.save_dir}")
        return self.performance_history, self.representations_history

    def _hnet_train_step(self, optimizer, trial, task_idx):
        """One optimisation step: generate weights → functional forward → task loss + reg."""
        self.model.train()
        self.hnet.train()
        optimizer.zero_grad()

        # Generate weights (differentiable — stays in computation graph)
        task_emb = self.task_embs[task_idx]
        params_dict = self.hnet.generate_params_dict(task_emb, self.param_info)

        # Functional forward: uses generated weights WITHOUT breaking the graph
        x, y, mask = trial.to_tensor(device=self.device)
        outputs = torch.func.functional_call(
            self.model, params_dict, (x,),
            kwargs={'return_all_states': False},
        )
        task_loss = compute_loss(outputs, y, mask,
                                 loss_type=trial.config.get('loss_type', 'cross_entropy'))

        # Output regularisation for previous tasks
        reg_loss = self._output_reg_loss(task_idx)
        loss = task_loss + self.hnet_beta * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.hnet.parameters()) + self.task_embs, max_norm=1.0
        )
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Evaluation (task-specific weight loading)
    # ------------------------------------------------------------------

    def _evaluate_hypernet(self, current_task_idx, current_task_name):
        """Evaluate all tasks, using each task's own embedding when available."""
        print(f"\n--- Evaluation after {current_task_name} ---")
        self.hnet.eval()

        for eval_idx, (eval_name, _) in enumerate(self.tasks):
            # Use the task's own embedding if learned, else current task's
            if eval_idx <= current_task_idx:
                emb = self.task_embs[eval_idx]
            else:
                emb = self.task_embs[current_task_idx]

            with torch.no_grad():
                self._load_generated_weights(emb)

            test_trial = self.fixed_test_sets[eval_name]
            metrics = evaluate_model(self.model, test_trial, device=self.device)
            self.performance_history[eval_name].append(metrics)
            acc_str = f", acc={metrics['accuracy']:.2%}" if not np.isnan(metrics['accuracy']) else ""
            print(f"  {eval_name}: loss={metrics['loss']:.4f}{acc_str}")

            reps = extract_rnn_representations(self.model, test_trial, device=self.device)
            self.representations_history[eval_name].append(reps.cpu().numpy())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_hnet_checkpoint(self, task_idx):
        """Save hypernetwork state + task embeddings."""
        import os
        hnet_dir = os.path.join(self.save_dir, "hnet_checkpoints")
        os.makedirs(hnet_dir, exist_ok=True)
        ckpt = {
            'hnet_state_dict': self.hnet.state_dict(),
            'task_embs': [e.data.cpu() for e in self.task_embs],
            'stored_targets': [t.cpu() for t in self.stored_targets],
        }
        path = os.path.join(hnet_dir, f"hnet_after_task_{task_idx}.pth")
        torch.save(ckpt, path)
        print(f"  HyperNet checkpoint saved: {path}")
