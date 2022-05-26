import sys
import os
import json
import torch
import numpy as np
import time


from opts import get_args
from utils.logger import Logger

def main(args):
    # system setup
    global CUDA_SUPPORT

    Logger.setup_logging(args.loglevel, logfile=args.logfile)
    logger = Logger()

    logger.debug(f"CLI args: {args}")

    if torch.cuda.device_count():
        CUDA_SUPPORT = True
    else:
        logger.warning('CUDA unsupported!!')
        CUDA_SUPPORT = False

    if not CUDA_SUPPORT:
        args.gpu = "cpu"

    if args.deterministic:
        # import torch.backends.cudnn as cudnn
        import os
        import random

        if CUDA_SUPPORT:
            # cudnn.deterministic = args.deterministic
            # cudnn.benchmark = not args.deterministic
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)

        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    logger.info(f"Model: {args.model}, Dataset:{args.dataset}")

    # In case of DataParallel for .to() to work
    args.device = args.gpu[0] if type(args.gpu) == list else args.gpu

    # Load data sets
    # trainsets, testset = load_data(args.data_path, args.dataset,
    #                                load_trainset=True, download=True)
    #
    # test_batch_size = get_test_batch_size(args.dataset, args.batch_size)
    # testloader = torch.utils.data.DataLoader(testset,
    #                                          batch_size=test_batch_size,
    #                                          num_workers=4,
    #                                          shuffle=False,
    #                                          persistent_workers=True)
    #
    # init_and_train_model(args, trainsets, testloader)



def init_and_train_model(args, trainsets, testloader):
   pass


if __name__ == "__main__":
    args = get_args(sys.argv)
    # run locally
    main(args)
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() == 0