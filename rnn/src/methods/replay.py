import random
import torch
from src.methods.base import BaseMethod
from src.train import compute_loss


class ReplayMethod(BaseMethod):
    """Experience Replay: interleave stored trials from previous tasks during training."""

    def __init__(self, memory_per_task=50, replay_num_tasks=1, **kwargs):
        super().__init__(**kwargs)
        self.memory_per_task = memory_per_task  # trials to keep per past task
        self.replay_num_tasks = replay_num_tasks  # past tasks to sample per step
        self.memory = {}  # task_name -> list of Trial objects

    def _replay_train_step(self, optimizer, trial, replay_trials):
        """Training step on current trial + replay trials."""
        self.model.train()
        optimizer.zero_grad()

        # Current task loss
        x, y, mask = trial.to_tensor(device=self.device)
        outputs = self.model(x, return_all_states=False)
        loss = compute_loss(outputs, y, mask, loss_type=trial.config.get('loss_type', 'cross_entropy'))

        # Replay loss from stored past-task trials
        for r_trial in replay_trials:
            rx, ry, rmask = r_trial.to_tensor(device=self.device)
            r_outputs = self.model(rx, return_all_states=False)
            r_loss = compute_loss(r_outputs, ry, rmask,
                                  loss_type=r_trial.config.get('loss_type', 'cross_entropy'))
            loss += r_loss
        loss /= (1 + len(replay_trials))  # average over current + replay

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    def before_task(self, task_idx, task_name, task_gen_fn):
        """Log replay memory state at the start of each task."""
        if self.memory:
            total_mem = sum(len(v) for v in self.memory.values())
            print(f"  Replay memory: {total_mem} trials from {len(self.memory)} past tasks")

    def _train_one_step(self, optimizer, train_pool, i):
        """Inject replay trials while keeping the base early-stopping loop."""
        trial = train_pool[i % self.train_pool_size]
        past_tasks = list(self.memory.keys())
        if past_tasks:
            # Sample a subset of past tasks (not all) for efficiency
            k = min(self.replay_num_tasks, len(past_tasks))
            sampled = random.sample(past_tasks, k)
            replay_trials = [random.choice(self.memory[pt]) for pt in sampled]
            return self._replay_train_step(optimizer, trial, replay_trials)
        return self.train_step(optimizer, trial)

    def after_task(self, task_idx, task_name, task_gen_fn):
        """Store a random subset of training trials into replay memory."""
        train_pool = self.fixed_train_sets[task_name]
        n_store = min(self.memory_per_task, len(train_pool))
        self.memory[task_name] = random.sample(train_pool, n_store)
        print(f"  Stored {n_store} trials from {task_name} into replay memory")
