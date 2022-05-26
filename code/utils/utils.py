import os
import numpy as np
import json
import glob
import torch

class BColors(object):
    HEADER = "\033[95m"
    OK_BLUE = "\033[94m"
    OK_CYAN = "\033[96m"
    OK_GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END_C = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def touch(fname: str, times=None, create_dirs: bool = False):
    if create_dirs:
        base_dir = os.path.dirname(fname)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    with open(fname, "a"):
        os.utime(fname, times)


def touch_dir(base_dir: str) -> None:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def top1_accuracy(output, target):
    return accuracy(output, target, topk=(1,))[0].item()

@torch.no_grad()
def grad_norm(output, target, model):
    flatten_grad = torch.cat([p.grad.view(-1) for p in model.parameters()
                                if p.grad is not None])
    grad_norm = torch.sum(flatten_grad**2)
    return grad_norm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else None

    def get_val(self):
        return self.val

    def get_avg(self):
        return self.avg


def init_metrics_meter(metrics_dict=None, round=None):
    if metrics_dict is None:
        metrics_dict = {}
    if round is not None:
        metrics_meter = {
            'round': round,
            'loss': AverageMeter(),
            **{key: AverageMeter() for key in metrics_dict.keys()}
        }
    else:
        metrics_meter = {
            'train_round': [], 'train_loss': [],
            **{f'train_{key}': [] for key in metrics_dict.keys()},
            'test_round': [], 'test_loss': [],
            **{f'test_{key}': [] for key in metrics_dict.keys()}
        }
    return metrics_meter



def get_model_str_from_obj(model):
    return str(list(model.modules())[0]).split("\n")[0][:-1]


def create_metrics_dict(metrics):
    metrics_dict = {'round': metrics['round']}
    for k in metrics:
        if k == 'round':
            continue
        metrics_dict[k] = metrics[k].get_avg()
    return metrics_dict


def create_model_dir(args, lr=True):
    run_id = f'id={args.run_id}'
    attack = f'attack={args.attack}'
    agg = f'agg={args.agg}_{args.bucketing}'
    model = f'model={args.model}'
    compression = f'compression={args.compression}'

    model_dir = os.path.join(
        args.outputs_dir, run_id, agg, attack, compression, model
    )
    if lr:
        run_hp = os.path.join(
            f"lr={str(args.lr)}",
            f"seed={str(args.seed)}")
        model_dir = os.path.join(model_dir, run_hp)

    return model_dir


def metric_to_dict(metrics_meter, metrics_dict, round, preffix='', all_prefix=True):
    round_preffix = preffix + '_round' if all_prefix else 'round'
    out = {
        round_preffix: round,
        preffix + '_loss': metrics_meter['loss'].get_avg(),
        **{
            preffix + f'_{key}': metrics_meter[key].get_avg() for key in metrics_dict
        }
    }
    return out


def extend_metrics_dict(full_metrics, last_metrics):
    for k in last_metrics:
        if last_metrics[k] is not None:
            full_metrics[k].append(last_metrics[k])


def get_key(train=True):
    return 'train_' if train else 'test_'


def get_best_lr_and_metric(args, last=True):
    best_arg, best_lookup = (np.nanargmin, np.nanmin) \
        if args.metric in ['loss'] else (np.nanargmax, np.nanmax)
    key = get_key(args.train_metric)
    model_dir_no_lr = create_model_dir(args, lr=False)
    lr_dirs = [lr_dir for lr_dir in os.listdir(model_dir_no_lr)
               if os.path.isdir(os.path.join(model_dir_no_lr, lr_dir))
               and not lr_dir.startswith('.')]
    runs_metric = list()
    for lr_dir in lr_dirs:
        # /*/ for different seeds
        lr_metric_dirs = glob.glob(
            model_dir_no_lr + '/' + lr_dir + '/*/full_metrics.json')
        if len(lr_metric_dirs) == 0:
            runs_metric.append(np.nan)
        else:
            lr_metric = list()
            for lr_metric_dir in lr_metric_dirs:
                with open(lr_metric_dir) as json_file:
                    metrics = json.load(json_file)
                metric_values = metrics[key + args.metric]
                metric = metric_values[-1] if last else \
                    best_lookup(metric_values)
                lr_metric.append(metric)
            runs_metric.append(np.mean(lr_metric))

    i_best_lr = best_arg(runs_metric)
    best_metric = runs_metric[i_best_lr]
    best_lr = lr_dirs[i_best_lr]
    return best_lr, best_metric, lr_dirs


def get_best_runs(args_exp, last=True):
    model_dir_no_lr = create_model_dir(args_exp, lr=False)
    best_lr, _, _ = get_best_lr_and_metric(args_exp, last=last)
    model_dir_lr = os.path.join(model_dir_no_lr, best_lr)
    json_dir = 'full_metrics.json'
    metric_dirs = glob.glob(model_dir_lr + '/*/' + json_dir)

    print(f'Best_lr: {best_lr}')
    with open(metric_dirs[0]) as json_file:
        metric = json.load(json_file)
    runs = [metric]

    for metric_dir in metric_dirs[1:]:
        with open(metric_dir) as json_file:
            metric = json.load(json_file)
        # ignores failed runs
        if not np.isnan(metric[get_key(train=True) + 'loss']).any():
            runs.append(metric)

    return runs
