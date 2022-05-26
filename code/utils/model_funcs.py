import numpy as np


def accuracy(output, label, topk=(1,)):
    """
    Extract the accuracy of the model.
    :param output: The output of the model
    :param label: The correct target label
    :param topk: Which accuracies to return (e.g. top1, top5)
    :return: The accuracies requested
    """
    maxk = max(topk)
    if maxk > output.shape[-1]:
        maxk = output.shape[-1]
        topk = (np.min([maxk, k]) for k in topk)
    batch_size = label.size(0)

    if len(output.size()) == 1:
        _, pred = output.topk(maxk, 0, True, True)
    else:
        _, pred = output.topk(maxk, 1, True, True)
    if pred.size(0) != 1:
        pred = pred.t()

    if pred.size() == (1,):
        correct = pred.eq(label)
    else:
        correct = pred.eq(label.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def update_metrics(metrics_meter, name, value, batch_size):
    metrics_meter[name].update(value, batch_size)
