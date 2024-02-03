import torch

from worker import ByzantineWorker


class ShiftBackAttacker(ByzantineWorker):
    def __init__(self, multiplier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gradient = None
        self.original_model = self._save_current_model()
        self.mult = multiplier

    def get_gradient(self):
        return self._gradient

    def _save_current_model(self):
        layers_to_save = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                layers_to_save.append(
                    p.data.detach().clone().view(-1))
        return torch.cat(layers_to_save)

    def omniscient_callback(self):
        # check if the majority of workers are Byzantine
        good_workers = 0
        byz_workers = 0
        if self.simulator.subset_int is None:
            subset_int = range(len(self.simulator.workers))
        else:
            subset_int = self.simulator.subset_int
        for i in subset_int:
            if isinstance(self.simulator.workers[i], ByzantineWorker):
                byz_workers += 1
            else:
                good_workers += 1
        byzantine_majority = byz_workers > good_workers

        if byzantine_majority:
            # shift back to the original model
            print("Shift back to the original model")
            lr = self.optimizer.param_groups[0]["lr"]
            if self.mult == 'lr':
                mult = lr
            else:
                mult = self.mult
            self._gradient = \
                mult / lr * (self._save_current_model() - self.original_model)
        else:
            # be a good worker, average the gradients
            good_grads = []
            for i, w in enumerate(self.simulator.workers):
                if not isinstance(w, ByzantineWorker):
                    good_grads.append(w.get_gradient())
            self._gradient = torch.stack(good_grads).mean(dim=0)

    def set_gradient(self, gradient) -> None:
        raise NotImplementedError

    def apply_gradient(self) -> None:
        raise NotImplementedError
