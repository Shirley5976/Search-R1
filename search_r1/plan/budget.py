"""GPU-hour budget accounting for the plan (implementation checklist item 9).

Only the training step's wall-clock counts (rollout incl. retrieval, old/ref log-prob, reward and
advantage, actor update, and later the reader). Evaluation and checkpointing are timed outside the
step and never added. GPU seconds = wall seconds * number of GPUs.
"""


class GpuBudget:

    def __init__(self, n_gpus, eval_every_gpu_h=-1, budget_gpu_h=None):
        self.n_gpus = int(n_gpus)
        self.eval_every_s = float(eval_every_gpu_h) * 3600 if eval_every_gpu_h and eval_every_gpu_h > 0 else None
        self.budget_s = float(budget_gpu_h) * 3600 if budget_gpu_h else None
        self.used_s = 0.0
        self.next_eval_s = self.eval_every_s

    @property
    def used_gpu_h(self):
        return self.used_s / 3600

    def add_step(self, step_wall_s):
        self.used_s += float(step_wall_s) * self.n_gpus

    def eval_due(self):
        """True once per crossed eval threshold; skipped thresholds collapse into one eval."""
        if self.next_eval_s is None or self.used_s < self.next_eval_s:
            return False
        self.next_eval_s = (int(self.used_s // self.eval_every_s) + 1) * self.eval_every_s
        return True

    def exhausted(self):
        return self.budget_s is not None and self.used_s >= self.budget_s

    def state_dict(self):
        return {'used_s': self.used_s, 'next_eval_s': self.next_eval_s}

    def load_state_dict(self, state):
        self.used_s = float(state['used_s'])
        self.next_eval_s = state['next_eval_s']
