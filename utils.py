import torch
import time
import numpy as np
import random
from logger import *
import json
import os

SEED = 1


def get_ms():
    """Returns the current time in milliseconds."""
    return time.time() * 1000


def init_seed(seed=None):
    global SEED
    """Seed the RNGs for predictability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)
    SEED = seed
    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num - 1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format("=" * (fill - 1) + ">", " " * (40 - fill), batch_num, last_loss), end='')


def save_checkpoint(net, name, batch_num, losses, costs, seq_lengths, repeats=1, checkpoint_path='./', seed=-1):
    progress_clean()
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    basename = "{}/{}-{}-batch-{}".format(checkpoint_path, name, seed, batch_num)
    model_fname = basename + ".model"
    print("Saving model checkpoint to: '{}'".format(model_fname))
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    print("Saving model training history to '{}'".format(train_fname))
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths,
        "repeat": repeats
    }
    open(train_fname, 'wt').write(json.dumps(content))


def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)
