import torch
import numpy as np
from torch.utils.data.sampler import Sampler


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, full_dataset=False, shuffle_iter=False):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.total_size = len(self.dataset)
        # self.num_samples = len(self.dataset)
        self.full_dataset = full_dataset
        if full_dataset:
            self.num_samples = self.total_size
        else:
            self.num_samples = (self.total_size - rank - 1) // self.num_replicas + 1
        self.shuffle = shuffle
        self.shuffle_iter = shuffle_iter

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(0)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # subsample
        if not self.full_dataset:
            indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples, (len(indices), self.num_samples)
        if self.shuffle_iter:
            idx = np.arange(self.num_samples)
            np.random.shuffle(idx)
            indices = list(np.array(indices)[idx])
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __str__(self):
        return "DistributedSampler(num_replicas={num_replicas},rank={rank},shuffle={shuffle})".format(
            num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle
        )
