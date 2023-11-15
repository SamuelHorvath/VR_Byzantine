import os
import sys
import json
import numpy as np
from copy import deepcopy
import torch
from torch import nn

from opts import get_args

# Utility functions
from data_funcs.libsvm import LibSVM
from tasks.libsvm import LogisticRegression, libsvm
from tasks.loss import Loss
from utils.utils import top1_accuracy, grad_norm, \
    create_model_dir, init_metrics_meter, extend_metrics_dict, metric_to_dict
from utils.logger import Logger
from utils.random_generator import RandomNumber
from compressors import get_compression

# Attacks
from attacks import *

# Main Modules
from worker import MomentumWorker, MarinaWorker, DianaWorker
from server import TorchServer
from simulator import ParallelTrainer, DistributedEvaluator

from utils.byz_funcs import get_sampler_callback, get_aggregator, get_test_sampler_callback

# Fixed HPs
# BATCH_SIZE = 32
# TEST_BATCH_SIZE = 1024

DATASET = 'a9a'
EXTRA_ID = ''
N_FEATURES = 123
PENALTY = 1e-2

def initialize_worker(
        args,
        trainer,
        worker_rank,
        model,
        model_snap,
        optimizer,
        optimizer_snap,
        loss_func,
        device,
        kwargs,
):
    compression = get_compression(args.compression)

    train_loader = libsvm(
        data_dir=args.data_path,
        name=DATASET,
        download=True,
        batch_size=args.batch_size,
        sampler_callback=get_sampler_callback(args, worker_rank),
        dataset_cls=LibSVM,
        drop_last=False,
        **kwargs,
    )

    if worker_rank < args.n - args.f:
        if args.model == 'marina':
            return MarinaWorker(
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                clip_update=args.clip_update,
                clip_mult=args.clip_mult,
                **kwargs,
            )
        elif args.model == 'diana':
            return DianaWorker(
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        elif args.model == 'mom_sgd':
            return MomentumWorker(
                momentum=0.9,
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        elif args.model == 'sgd':
            return MomentumWorker(
                momentum=0.,
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

    if args.attack == "BF":
        attacker = BitFlippingWorker(
            compression=compression,
            data_loader=train_loader,
            model=model,
            model_snap=model_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            optimizer_snap=optimizer_snap,
            **kwargs,
        )
        attacker.configure(trainer)
        return attacker

    if args.attack == "LF":
        attacker =  LableFlippingWorker(
            revertible_label_transformer=lambda y: 1 - y,
            compression=compression,
            data_loader=train_loader,
            model=model,
            model_snap=model_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            optimizer_snap=optimizer_snap,
            **kwargs,
        )
        attacker.configure(trainer)
        return attacker

    if args.attack == "mimic":
        attacker = MimicVariantAttacker(
            compression=compression,
            warmup=args.mimic_warmup,
            data_loader=train_loader,
            model=model,
            model_snap=model_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            optimizer_snap=optimizer_snap,
            **kwargs,
        )
        attacker.configure(trainer)
        return attacker

    if args.attack == "IPM":
        attacker = IPMAttack(
            compression=compression,
            epsilon=0.1,
            data_loader=train_loader,
            model=model,
            model_snap=model_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            optimizer_snap=optimizer_snap,
            **kwargs,
        )
        attacker.configure(trainer)
        return attacker

    if args.attack == "ALIE":
        attacker = ALittleIsEnoughAttack(
            n=args.n,
            m=args.f,
            # z=1.5,
            compression=compression,
            data_loader=train_loader,
            model=model,
            model_snap=model_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            optimizer_snap=optimizer_snap,
            **kwargs,
        )
        attacker.configure(trainer)
        return attacker

    raise NotImplementedError(f"No such attack {args.attack}")


def main(args):
    # initialize_logger(args.logfile)
    Logger.setup_logging(args.loglevel, logfile=args.logfile)
    Logger()

    args.run_id = DATASET + EXTRA_ID
    if args.full_dataset:
        args.run_id += '_full'

    if args.use_cuda and not torch.cuda.is_available():
        print("=> There is no cuda device!!!!")
        device = "cpu"
    else:
        device = torch.device("cuda" if args.use_cuda else "cpu")
    # kwargs = {"num_workers": 1, "pin_memory": True} if args.use_cuda else {}
    kwargs = {"pin_memory": True, "num_workers": args.num_workers} if args.use_cuda else {}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = LogisticRegression(N_FEATURES).to(device)
    model_snap_s = [deepcopy(model) for _ in range(args.n)]

    # Each optimizer contains a separate `state` to store info like `momentum_buffer`
    optimizers = [torch.optim.SGD(model.parameters(), lr=args.lr) for _ in range(args.n)]
    optimizers_snap = [torch.optim.SGD(model_snap_s[i].parameters(), lr=args.lr) for i in range(args.n)]
    server_opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    Loss(nn.BCEWithLogitsLoss(), True, PENALTY)
    loss_func = Loss.compute_loss

    metrics = {}

    if args.batch_size > args.test_batch_size:
        args.test_batch_size = args.batch_size

    server = TorchServer(optimizer=server_opt)
    trainer = ParallelTrainer(
        server=server,
        aggregator=get_aggregator(args),
        max_batches_per_epoch=args.max_batches,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,

    )

    train_loader = libsvm(
        data_dir=args.data_path,
        name=DATASET,
        download=True,
        batch_size=args.test_batch_size,
        shuffle=False,
        sampler_callback=get_test_sampler_callback(args),
        **kwargs,
    )

    train_evaluator = DistributedEvaluator(
        model=model,
        data_loader=train_loader,
        loss_func=loss_func,
        device=device,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
        log_identifier_type='train',
    )

    if args.attack == "NA":
        args.n -= args.f
        args.f = 0

    for worker_rank in range(args.n):
        worker = initialize_worker(
            args,
            trainer,
            worker_rank,
            model=model,
            model_snap=model_snap_s[worker_rank],
            optimizer=optimizers[worker_rank],
            optimizer_snap=optimizers_snap[worker_rank],
            loss_func=loss_func,
            device=device,
            kwargs={},
        )
        trainer.add_worker(worker)

    # RandomNumber.full_grad_prob = 1 / len(trainer.workers[0].data_loader)
    RandomNumber.full_grad_prob = 0.

    if not args.dry_run:
        full_metrics = init_metrics_meter(metrics)
        model_dir = create_model_dir(args)
        if os.path.exists(os.path.join(
                model_dir, 'full_metrics.json')):
            Logger.get().info(f"{model_dir} already exists.")
            Logger.get().info("Skipping this setup.")
            return
        # create model directory
        os.makedirs(model_dir, exist_ok=True)
        train_metric = train_evaluator.evaluate(0)
        extend_metrics_dict(
            full_metrics, metric_to_dict(train_metric, metrics, 0, 'train'))
        for epoch in range(1, args.epochs + 1):
            RandomNumber.full_grad = True
            trainer.train(
                epoch, args.partial_participation,
                args.partial_participation_ratio)
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                train_metric = train_evaluator.evaluate(epoch)
                extend_metrics_dict(
                    full_metrics, metric_to_dict(train_metric, metrics, epoch, 'train'))
            trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))
        #  store the run
        with open(os.path.join(
                model_dir, 'full_metrics.json'), 'w') as f:
            json.dump(full_metrics, f, indent=4)


if __name__ == "__main__":
    args = get_args(sys.argv)
    main(args)
    torch.cuda.empty_cache()
    # assert torch.cuda.memory_allocated() == 0
