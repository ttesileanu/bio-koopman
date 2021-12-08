""" Define various miscellaneous utilities. """

import numpy as np

from typing import Sequence


class StepwiseScheduler:
    def __init__(self, optimizer, sequence: Sequence):
        assert len(optimizer.param_groups) == 1

        self.optimizer = optimizer
        self.sequence = np.copy(sequence)
        self._borders = np.cumsum([_[0] for _ in self.sequence])

        self._step_count = 0
        self._last_lr = None

        self.step()

    def get_last_lr(self) -> list:
        return self._last_lr

    def get_lr(self) -> list:
        mask = self._step_count <= self._borders
        if not np.any(mask):
            lr = [self.sequence[-1][1]]
        else:
            idx = mask.nonzero()[0][0]
            lr = [self.sequence[idx][1]]

        return lr

    def step(self):
        self._step_count += 1
        lr = self.get_lr()
        for group, crt_lr in zip(self.optimizer.param_groups, lr):
            group["lr"] = crt_lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
