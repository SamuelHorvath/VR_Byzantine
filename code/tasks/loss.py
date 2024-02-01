import torch


class Loss:
    criterion = None
    regularizer = False
    regularizer_value = 1e-2

    def __init__(self, criterion, regularizer=False, regularizer_value=1e-2):
        Loss.criterion = criterion
        Loss.regularizer = regularizer
        Loss.regularizer_value = regularizer_value

    @classmethod
    def compute_loss(cls, output, label, model):
        loss = cls.criterion(output, label)
        if Loss.regularizer:
            flatten_params = torch.cat([p.view(-1) for p in model.parameters()])
            penalty = (cls.regularizer_value / 2) * torch.sum(flatten_params**2)
            loss += penalty
        return loss
