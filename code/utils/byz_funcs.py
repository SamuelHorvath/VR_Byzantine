import numpy as np

# IID vs Non-IID
from data_funcs.sampler import (
    DistributedSampler,
)

# Aggregators
from aggregator import *


def _get_aggregator(args):
    if args.agg == "avg":
        return Mean()

    if args.agg == "cm":
        return CM()

    if args.agg == "cp":
        if args.clip_scaling is None:
            tau = args.clip_tau
        elif args.clip_scaling == "linear":
            tau = args.clip_tau / (1 - args.momentum)
        elif args.clip_scaling == "sqrt":
            tau = args.clip_tau / np.sqrt(1 - args.momentum)
        else:
            raise NotImplementedError(args.clip_scaling)
        return Clipping(tau=tau, n_iter=3)

    if args.agg == "rfa":
        return RFA(T=8)

    if args.agg == "tm":
        return TM(b=args.f)

    if args.agg == "krum":
        T = int(np.ceil(args.n / args.bucketing)) if args.bucketing > 0 else args.n
        return Krum(n=T, f=args.f, m=1)

    raise NotImplementedError(args.agg)


def bucketing_wrapper(args, aggregator, s):
    """
    Key functionality.
    """
    print("Using bucketing wrapper.")

    def aggr(inputs):
        n = len(inputs)
        indices = list(range(n))
        np.random.shuffle(indices)

        T = int(np.ceil(n / s))

        reshuffled_inputs = []
        for t in range(T):
            indices_slice = indices[t * s : (t + 1) * s]
            g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
            reshuffled_inputs.append(g_bar)
        return aggregator(reshuffled_inputs)

    return aggr


def get_aggregator(args):
    aggr = _get_aggregator(args)
    if args.bucketing == 0:
        return aggr

    return bucketing_wrapper(args, aggr, args.bucketing)


def get_sampler_callback(args, rank, shuffle=True):
    """
    Get sampler based on the rank of a worker.
    The first `n-f` workers are good, and the rest are Byzantine
    """
    n_good = args.n - args.f
    if rank >= n_good:
        # Byzantine workers
        return lambda x: DistributedSampler(
            num_replicas=n_good,
            rank=rank % n_good,
            shuffle=shuffle,
            dataset=x,
            full_dataset=args.full_dataset,
            shuffle_iter=True,
        )


    return lambda x: DistributedSampler(
        num_replicas=n_good,
        rank=rank,
        shuffle=shuffle,
        dataset=x,
        full_dataset=args.full_dataset,
        shuffle_iter=True,
    )


def get_test_sampler_callback(args):
    # This alpha argument is not important as there is
    # only 1 replica
    # return lambda x: NONIIDLTSampler(
    #     alpha=True,
    #     beta=0.5 if args.LT else 1.0,
    #     num_replicas=1,
    #     rank=0,
    #     shuffle=False,
    #     dataset=x,
    # )

    return lambda x: DistributedSampler(
        num_replicas=1,
        rank=0,
        shuffle=False,
        dataset=x,
        shuffle_iter=False,
    )
