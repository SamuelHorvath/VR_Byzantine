import numpy as np
import torch
from typing import Union, Callable, Any, List

from utils.logger import Logger
from utils.utils import init_metrics_meter
from utils.model_funcs import update_metrics
from utils.random_generator import RandomNumber

from worker import TorchWorker
from server import TorchServer


class DistributedSimulatorBase(object):
    """Simulate distributed programs with low memory usage.
    Functionality:
    1. randomness control: numpy, torch, torch-cuda
    2. add workers
    This base class is used by both trainer and evaluator.
    """

    def __init__(self, metrics: dict, use_cuda: bool, debug: bool):
        """
        Args:
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.metrics = metrics
        self.use_cuda = use_cuda
        self.debug = debug
        self.workers = []

        # self.json_logger = Logger.get()
        self.debug_logger = Logger.get()


class ParallelTrainer(DistributedSimulatorBase):
    """Synchronous and parallel training with specified aggregator."""

    def __init__(
            self,
            server: TorchServer,
            aggregator: Callable[[list], torch.Tensor],
            max_batches_per_epoch: int,
            log_interval: int,
            metrics: dict,
            use_cuda: bool,
            debug: bool,
    ):
        """
        Args:
            aggregator (callable): A callable which takes a list of tensors
            and returns an aggregated tensor.
            max_batches_per_epoch (int): Set the maximum number of batches
              in an epoch.
                Usually used for debugging.
            log_interval (int): Control the frequency of logging
            training batches
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.aggregator = aggregator
        self.server = server
        self.log_interval = log_interval
        self.max_batches_per_epoch = max_batches_per_epoch
        self.omniscient_callbacks = []
        self.random_states = {}
        super().__init__(metrics, use_cuda, debug)
        self.subset_int = None  # for partial participation

    def aggregation_and_update(self,
                               clip_update: bool = False,
                               clip_mult: float = 1.0):
        # If there are Byzantine workers,
        # ask them to craft attacks based on the updated models.
        # Attackers know all the clients
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback()

        # print("subset", subset)
        gradients = self.parallel_get(
            lambda w: w.get_gradient(), subset=self.subset_int)

        # for i in range(len(gradients)):
        #     print(f"Gradient {i} norm {torch.norm(gradients[i])}")
        #     break

        # print('Gradients len:', len(gradients))
        # print('Gradients stat:',
        #     torch.std(torch.stack(gradients, dim=0), dim=0))
        if clip_update:
            if self.server.last_gradient is None:
                raise ValueError("No last gradient to clip")

            # compute the threshold: clip_mult * ||x_k - x_{k-1}||_2
            # lr = self.server.optimizer.param_groups[0]['lr']
            threshold = clip_mult * torch.norm(self.server.last_gradient)
            # print(f"Threshold {threshold:.6f}")

            def clip_update(gradient):
                norm = torch.norm(gradient)
                if norm > threshold:
                    # print('Clipping gradient')
                    new_gradient = gradient * (threshold / norm)
                    # assert torch.allclose(
                    #     torch.norm(new_gradient), threshold)
                    return new_gradient

                return gradient

            # shift and clip client gradients
            shifted_gradients = [
                clip_update(gradient - self.server.last_gradient)
                for gradient in gradients]
            # shift back and aggregate
            aggregated = self.server.last_gradient + self.aggregator(
                shifted_gradients)
            # aggregated_no_clip = self.aggregator(gradients)
            # print(torch.norm(aggregated - aggregated_no_clip))
        else:
            aggregated = self.aggregator(gradients)
        # print(f"Global gradient norm {torch.norm(aggregated)}")

        # each client know the "global gradient", i.e., aggregated
        self.parallel_call(lambda w: w.set_global_gradient(aggregated))

        # Assume that the model and optimizers are shared among workers.
        self.server.set_gradient(aggregated)
        self.server.apply_gradient()

    def train(self, epoch, partial_participation=False,
              partial_participation_ratio=0.2,
              clip_update=False, clip_mult=1.0):
        self.debug_logger.info(f"Train epoch {epoch}")
        self.parallel_call(lambda worker: worker.train_epoch_start())

        metrics_meter = init_metrics_meter(self.metrics, epoch)
        for batch_idx in range(self.max_batches_per_epoch):
            try:
                # partial participation
                if partial_participation and not RandomNumber.full_grad:
                    self.subset_int = np.random.choice(
                        len(self.workers),
                        int(len(self.workers) * partial_participation_ratio),
                        replace=False,
                    )
                else:
                    self.subset_int = None

                # compute gradient for all the clients,
                # to be used by omniscient attackers
                results = self.parallel_get(
                    lambda w: w.compute_gradient(), subset=None)
                # aggregate the gradients of clients that participate
                self.aggregation_and_update(
                    # do not clip when full_grad,
                    # i.e., full participation, is True
                    clip_update=clip_update and not RandomNumber.full_grad,
                    clip_mult=clip_mult)

                for res in results:
                    update_metrics(
                        metrics_meter, 'loss', res['loss'], res['batch_size'])
                    for key in self.metrics:
                        update_metrics(
                            metrics_meter, key, res["metrics"][key],
                            res['batch_size'])
                if batch_idx % self.log_interval == 0:
                    self.log_train(metrics_meter, batch_idx, epoch)
                RandomNumber.sample()
            except StopIteration:
                break
        return metrics_meter

    # ---------------------------------------------------------------------------- #
    #                                    Utility                                   #
    # ---------------------------------------------------------------------------- #

    def add_worker(self, worker: TorchWorker):
        worker.add_metrics(self.metrics)
        self.workers.append(worker)
        self.debug_logger.info(f"=> Add worker {worker}")

    def register_omniscient_callback(self, callback):
        self.omniscient_callbacks.append(callback)

    def cache_random_state(self) -> None:
        if self.use_cuda:
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()

    def restore_random_state(self) -> None:
        if self.use_cuda:
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])

    def parallel_call(self, f: Callable[[TorchWorker], None],
                      subset: List[int] = None) -> None:
        if subset is None:
            subset_workers = self.workers
        else:
            subset_workers = [self.workers[i] for i in subset]

        for w in subset_workers:
            self.cache_random_state()
            f(w)
            self.restore_random_state()

    def parallel_get(self, f: Callable[[TorchWorker], Any],
                     subset: List[int] = None) -> list:
        if subset is None:
            subset_workers = self.workers
        else:
            subset_workers = [self.workers[i] for i in subset]

        results = []
        for w in subset_workers:
            self.cache_random_state()
            results.append(f(w))
            self.restore_random_state()
        return results

    # ---------------------------------------------------------------------------- #
    #                                Log information                               #
    # ---------------------------------------------------------------------------- #

    def __str__(self):
        return (
            "ParallelTrainer("
            f"aggregator={self.aggregator}, "
            f"max_batches_per_epoch={self.max_batches_per_epoch}, "
            f"log_interval={self.log_interval}, "
            f"metrics={list(self.metrics.keys())}"
            f"use_cuda={self.use_cuda}, "
            f"debug={self.debug}, "
            ")"
        )

    def log_train(self, metrics_meter, batch_idx, epoch):

        # Output to console
        self.debug_logger.info(
            f"Epoch: {epoch :2} Batch: {batch_idx}| {len(self.workers[0].data_loader)}|"
            f"  Loss: {metrics_meter['loss'].get_avg():.4f} "
            + " ".join(key + "=" + "{:>8.4f}".format(metrics_meter[key].get_avg()) for key in self.metrics)
        )

        # Output to file
        # self.json_logger.info(r)


class DistributedEvaluator(DistributedSimulatorBase):
    def __init__(
            self,
            model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            loss_func: torch.nn.modules.loss._Loss,
            device: Union[torch.device, str],
            metrics: dict,
            use_cuda: bool,
            debug: bool,
            log_identifier_type="validation",
    ):
        super().__init__(metrics, use_cuda, debug)
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.device = device
        self.log_identifier_type = log_identifier_type

    def __str__(self):
        return (
            "DistributedEvaluator("
            f"use_cuda={self.use_cuda}, "
            f"debug={self.debug}, "
            ")"
        )

    def evaluate(self, epoch):
        self.model.eval()
        metrics_meter = init_metrics_meter(self.metrics, epoch)

        # total_loss = 0
        # n_points = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.data_loader):
                batch_size = data.shape[0]
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_func(output, target, self.model).item()
                # total_loss += loss * batch_size
                # n_points += batch_size
                update_metrics(metrics_meter, 'loss', loss, batch_size)
                for key in self.metrics:
                    update_metrics(metrics_meter, key, self.metrics[key](output, target), batch_size)
        # print(n_points)
        # total_loss /= n_points
        # print(total_loss)
        # Output to file
        self.debug_logger.info(
            f" {self.log_identifier_type} loss = {metrics_meter['loss'].get_avg():.4f}; "
            + " ".join(key + " = " + "{}".format(metrics_meter[key].get_avg()) for key in self.metrics)
        )
        return metrics_meter
