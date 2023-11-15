import argparse
import time
from datetime import datetime
import os


def get_args(args):
    parser = initialise_arg_parser(args, 'Variance Reduced Byzantine.')

    # Utility
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--identifier", type=str, default="debug", help="")

    # Experiment configuration
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-m", "--max-batches", type=int, default=1000, help="Maximal number of batches per epoch")
    parser.add_argument("-n", type=int, help="Number of workers")
    parser.add_argument("-f", type=int, help="Number of Byzantine workers.")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--test-batch-size", default=128, type=int, help="Test batch size")
    parser.add_argument("--full-dataset", action="store_true", default=False)

    # Partial participation
    parser.add_argument("--partial-participation", action="store_true", default=False)
    parser.add_argument("--partial-participation-ratio", type=float, default=0.2)

    # Clipping
    parser.add_argument("--clip-update", action="store_true", default=False)
    parser.add_argument("--clip-mult", type=float, default=2.)

    parser.add_argument("--attack", type=str, default="NA", help="Type of attacks.")
    parser.add_argument("--agg", type=str, default="avg", help="")
    parser.add_argument("--model", type=str, default="grad", help="")
    parser.add_argument("--compression", type=str, default="none", help="")
    parser.add_argument(
        "--noniid",
        action="store_true",
        default=False,
        help="[HP] noniidness.",
    )
    parser.add_argument("--LT", action="store_true", default=False, help="Long tail")

    # Key hyperparameter
    parser.add_argument("--bucketing", type=int, default=0, help="[HP] s")
    # parser.add_argument("--momentum", type=float, default=0.0, help="[HP] momentum")

    parser.add_argument("--clip-tau", type=float, default=10.0, help="[HP] clip tau")
    parser.add_argument("--clip-scaling", type=str, default=None, help="[HP] clip scaling")

    parser.add_argument(
        "--mimic-warmup", type=int, default=1, help="the warmup phase in iterations."
    )

    # SETUP ARGUMENTS
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="../outputs/",
        help="Base root directory for the dataset."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/",
        help="Base root directory for the dataset."
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Num workers for dataset loading"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="How often to do validation."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=str(time.time()),
        help="Identifier for the current job"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        default="INFO"
    )
    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")
    os.makedirs("../logs/", exist_ok=True)
    parser.add_argument(
        "--logfile",
        type=str,
        default=f"../logs/log_{now}.txt"
    )

    args = parser.parse_args()

    if args.n <= 0 or args.f < 0 or args.f >= args.n:
        raise RuntimeError(f"n={args.n} f={args.f}")

    assert args.bucketing >= 0, args.bucketing
    # assert args.momentum >= 0, args.momentum
    assert len(args.identifier) > 0
    return args


def initialise_arg_parser(args, description):
    parser = argparse.ArgumentParser(args, description=description)
    return parser
