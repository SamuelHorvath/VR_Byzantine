import torch
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Union, Callable, Tuple

from utils.random_generator import RandomNumber
from compressors import Identity, Compressor


class TorchWorker(object):
    """A worker for distributed training.
    Compute gradients locally and store the gradient.
    """

    def __init__(
            self,
            data_loader: torch.utils.data.DataLoader,
            model: torch.nn.Module,
            model_snap: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            optimizer_snap: torch.optim.Optimizer,
            loss_func: torch.nn.modules.loss._Loss,
            device: Union[torch.device, str],
            compression: Compressor,
    ):
        self.data_loader = data_loader
        # BUG: we would need scalar for non-uniform data split
        # self.scalar = len(data_loader.dataset) / (
        #         data_loader.sampler.num_replicas * len(data_loader.sampler))
        self.scalar = 1.
        self.model = model
        self.model_snap = model_snap
        self.optimizer = optimizer
        self.optimizer_snap = optimizer_snap
        self.loss_func = loss_func
        self.device = device
        if compression is None:
            self.compression = Identity()
        else:
            self.compression = compression

        # self.running has attribute:
        #   - `train_loader_iterator`: data iterator
        #   - `data`: last data
        #   - `target`: last target
        self.running = {}
        self.metrics = {}
        self.state = defaultdict(dict)
        self.global_gradient = None

    def add_metric(
            self,
            name: str,
            callback: Callable[[torch.Tensor, torch.Tensor], float],
    ):
        """
        The `callback` function takes predicted and groundtruth value
        and returns its metric.
        """
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")

        self.metrics[name] = callback

    def add_metrics(self, metrics: dict):
        for name in metrics:
            self.add_metric(name, metrics[name])

    def __str__(self) -> str:
        return "TorchWorker"

    def train_epoch_start(self) -> None:
        self.running["train_loader_iterator"] = iter(self.data_loader)
        self.model.train()

    def compute_gradient(self):
        results = {}

        data, target = self.running["train_loader_iterator"].__next__()
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_func(output, target, self.model)
        loss.backward()

        self.running["data"] = data
        self.running["target"] = target

        self._save_grad()

        self.model_snap.load_state_dict(deepcopy(self.model.state_dict()))

        results["loss"] = loss.item()
        results["batch_size"] = len(target)
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric(output, target)

        return results

    def get_gradient(self) -> torch.Tensor:
        return self.scalar * self._get_saved_grad()

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for p in self.model.parameters():
            end = beg + len(p.grad.view(-1))
            x = gradient[beg:end].reshape_as(p.grad.data)
            p.grad.data = x.clone().detach()
            beg = end

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                param_state["saved_grad"] = p.grad.data.detach().clone()

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(
                    self.compression(param_state["saved_grad"].data.view(-1)))
        return torch.cat(layer_gradients)

    def set_global_gradient(self, global_gradient: torch.Tensor) -> None:
        self.global_gradient = global_gradient


class MomentumWorker(TorchWorker):
    def __init__(self, momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = p.grad.data.detach().clone()
                else:
                    param_state["momentum_buffer"].mul_(self.momentum).add_(p.grad)

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(
                    self.compression(param_state["momentum_buffer"].data.view(-1)))
        return torch.cat(layer_gradients)

    def __str__(self) -> str:
        return "MomentumWorker"


class DianaWorker(TorchWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "shift_buffer" not in param_state:
                    param_state["shift_buffer"] = torch.zeros_like(p.grad.data.detach().clone())
                    param_state["grad_buffer"] = self.compression(p.grad.data.detach().clone())
                else:
                    diff = self.compression(
                        p.grad.data.detach().clone() - param_state["shift_buffer"])
                    param_state["grad_buffer"] = param_state["shift_buffer"] + diff
                    param_state["shift_buffer"].add_(diff, alpha=1 / self.compression.w)

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(
                    param_state["grad_buffer"].data.view(-1))
        return torch.cat(layer_gradients)

    def __str__(self) -> str:
        return "DianaWorker"


class MarinaWorker(TorchWorker):
    def __init__(self, *args, **kwargs):
        # self.clip_update = kwargs.pop("clip_update", False)
        # self.clip_mult = kwargs.pop("clip_mult", 2.)
        super().__init__(*args, **kwargs)

    def _compute_full_grad(self) -> None:
        self.optimizer.zero_grad()
        loss = 0.
        n_points = 0
        for data, target in self.data_loader:
            batch_size = data.shape[0]
            n_points += batch_size
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss += self.loss_func(output, target, self.model) * batch_size
        loss /= n_points
        loss.backward()

    def _compute_previous_grad(self) -> None:
        self.model_snap.train()
        self.optimizer_snap.zero_grad()
        data, target = self.running["data"], self.running["target"]
        output = self.model_snap(data)
        loss = self.loss_func(output, target, self.model_snap)
        loss.backward()

    def _save_grad(self) -> None:
        if RandomNumber.full_grad:
            self._compute_full_grad()
        else:
            # set current gradient to global gradient
            # and compute the gradient in prior point
            self._set_global_gradient_to_buffer()
            self._compute_previous_grad()

        for group, group_snap in zip(self.optimizer.param_groups,
                                     self.optimizer_snap.param_groups):
            for p, p_snap in zip(group["params"], group_snap["params"]):
                if p.grad is None:
                    continue
                param_state = self.state[p]
                # param_state["marina_buffer"] = p.grad.data.detach().clone()
                if RandomNumber.full_grad:
                    param_state["marina_buffer"] = p.grad.data.detach().clone()
                else:
                    diff = self.compression(
                        p.grad.data.detach().clone() -
                        p_snap.grad.data.detach().clone()
                    )

                    # # clipping
                    # clip_const = self.clip_mult * self.last_update_norm
                    # if self.clip_update:
                    #     if torch.norm(diff) > clip_const:
                    #         diff = diff / torch.norm(diff) * clip_const

                    param_state["marina_buffer"].add_(diff)

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["marina_buffer"].data.view(-1))
        grad = torch.cat(layer_gradients)
        # print("norm of marina gradient: ", torch.norm(grad))
        return grad

    def _set_global_gradient_to_buffer(self) -> None:
        gradient = self.global_gradient
        # print("shape of global gradient: ", gradient.shape)
        # print("norm of global gradient: ", torch.norm(gradient))
        # gradient = self._get_saved_grad()
        # print("shape of local gradient: ", gradient.shape)
        # print("norm of local gradient: ", torch.norm(gradient))
        if gradient is None:
            raise ValueError("Global gradient is not set.")
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                end = beg + len(p.grad.view(-1))
                x = gradient[beg:end].reshape_as(p.grad.data)
                param_state["marina_buffer"] = x.clone().detach()
                beg = end

    def __str__(self) -> str:
        return "MarinaWorker"


class ByzantineWorker(TorchWorker):
    def configure(self, simulator):
        # call configure after defining DistribtuedSimulator
        self.simulator = simulator
        simulator.register_omniscient_callback(self.omniscient_callback)

    def get_gradient(self):
        return self._gradient

    def omniscient_callback(self, subset: Optional[list] = None):
        # Loop over good workers and accumulate their gradients
        # only through the ones that participated in current round
        if subset is not None:
            subset_workers = [self.simulator.workers[i] for i in subset]
        else:
            subset_workers = self.simulator.workers

        gradients = []
        for w in subset_workers:
            if not isinstance(w, ByzantineWorker):
                gradients.append(w.get_gradient())

        # if no good workers participated, use the last gradient
        if len(gradients) > 0:
            stacked_gradients = torch.stack(gradients, 1)
            self._gradient = torch.mean(stacked_gradients, 1)

    def compute_gradient(self) -> Tuple[float, int]:
        # Use self.simulator to get all other workers
        # Note that the byzantine worker does not modify the states directly.
        return super().compute_gradient()

    # def get_gradient(self) -> torch.Tensor:
    #     # Use self.simulator to get all other workers
    #     return super().get_gradient()

    # def omniscient_callback(self):
    #     raise NotImplementedError

    def __str__(self) -> str:
        return "ByzantineWorker"
