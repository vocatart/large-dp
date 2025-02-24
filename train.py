import argparse
import os

from dp.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer
from dp.train import train
from torch import multiprocessing as mp
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper script for training DeepPhonemizer models.')
    parser.add_argument("exp_name", type=str, help="Experiment name")
    parser.add_argument("-c", "--checkpoint", action="store", help="Checkpoint file to resume from")
    args = parser.parse_args()

    # the newest ver of torch does not like some of the DP stuff
    torch.serialization.add_safe_globals([Preprocessor])
    torch.serialization.add_safe_globals([LanguageTokenizer])
    torch.serialization.add_safe_globals([SequenceTokenizer])

    num_gpus = torch.cuda.device_count()
    config_file = os.path.join("experiments", args.exp_name, "config.yaml")

    if args.checkpoint is not None:
        print(f"Resuming training from checkpoint {os.path.basename(args.checkpoint)}\n")

        if num_gpus > 1:
            mp.spawn(train, nprocs=num_gpus, args=(num_gpus, config_file, args.checkpoint))
        else:
            train(rank=0, num_gpus=num_gpus, config_file=config_file, checkpoint_file=args.checkpoint)
    else:
        if num_gpus > 1:
            mp.spawn(train, nprocs=num_gpus, args=(num_gpus, config_file))
        else:
            train(rank=0, num_gpus=num_gpus, config_file=config_file)
