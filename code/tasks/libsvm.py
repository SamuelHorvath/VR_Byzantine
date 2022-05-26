import torch
from torch import nn

from data_funcs.libsvm import LibSVM


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out


def libsvm(
        data_dir,
        name,
        download,
        batch_size,
        shuffle=None,
        sampler_callback=None,
        dataset_cls=LibSVM,
        drop_last=False,
        **loader_kwargs
):
    # if sampler_callback is not None and shuffle is not None:
    #     raise ValueError

    dataset = dataset_cls(
        data_dir,
        dataset_name=name,
        download=download,
    )

    sampler = sampler_callback(dataset) if sampler_callback else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        **loader_kwargs,
    )