import torch
import numpy as np

from .base_class import Compressor


class RandomSparsifier(Compressor):
    def __init__(self, h=0.1):
        self.h = h
        assert 0 < h <= 1, f"{h} out of boundaries"
        super().__init__(1 / h)

    def compress(self, x):
        return self._random_spars(x)

    def _random_spars(self, x):
        if not isinstance(x, torch.Tensor):
            return x

        y = x.view(-1)
        d = x.nelement()
        r = int(np.ceil(self.h * d))
        mask = np.random.choice(d, r, replace=False)
        y[mask] = 0
        return x * (d / r)
