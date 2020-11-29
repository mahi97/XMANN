#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Training for the Copy Task in Neural Turing Machines."""

import re
import sys

import attr

import numpy as np

from utils import *
from logger import *

from torch.nn.parallel import data_parallel

from tasks.tasks import TASKS


def update_model_params(params, update):
    """Updates the default parameters using supplied user arguments."""

    update_dict = {}
    for p in update:
        m = re.match("(.*)=(.*)", p)
        if not m:
            LOGGER.error("Unable to parse param update '%s'", p)
            sys.exit(1)

        k, v = m.groups()
        update_dict[k] = v

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)

    return params


def init_model(args):
    LOGGER.info("Training for the **%s** task", args.task)
    task = TASKS[args.task]
    params = task.param()
    params = update_model_params(params, args.param)

    LOGGER.info(params)

    model = task.model(params=params)
    return model


def evaluate(net, criterion, X, Y, is_cuda=False):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    states = []

    for i in range(inp_seq_len):
        if is_cuda:
            o, state = data_parallel(net, X[i])
        else:
            o, state = net(X[i])
        states += [state]

    if is_cuda:
        X = X.cuda()
        Y = Y.cuda()

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    if is_cuda:
        y_out = y_out.cuda()

    for i in range(outp_seq_len):
        if is_cuda:
            y_out[i], state = data_parallel(net, None)
        else:
            y_out[i], state = net()
        states += [state]

    loss = criterion(y_out, Y)

    if is_cuda:
        y_out_binarized = y_out.cpu().data
    else:
        y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.cpu().data))

    result = {
        'loss': loss.item(),
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }

    return result


def train_batch(net, criterion, optimizer, X, Y, is_cuda=False):
    """Trains a single batch."""
    optimizer.zero_grad()
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        if is_cuda:
            data_parallel(net, X[i])
        else:
            net(X[i])

    if is_cuda:
        X = X.cuda()
        Y = Y.cuda()

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    for i in range(outp_seq_len):
        if is_cuda:
            y_out[i], _ = data_parallel(net, None)
        else:
            y_out[i], _ = net()

    if is_cuda:
        Y = Y.cuda()
        y_out = y_out.cuda()

    loss = criterion(y_out, Y)
    loss.backward()
    clip_grads(net)
    optimizer.step()

    if is_cuda:
        y_out_binarized = y_out.cpu().data
    else:
        y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.cpu().data))

    return loss.item(), cost.item() / batch_size


def train_model(model, args):
    report_interval = args.report_interval
    checkpoint_interval = args.checkpoint_interval
    checkpoint_path = args.checkpoint_path
    is_cuda = args.GPU
    batch_size = model.params.batch_size

    losses = []
    costs = []
    seq_lengths = []
    repeats = []
    start_ms = get_ms()
    for batch_num, x, y in model.dataloader:
        if is_cuda:
            x = x.cuda()
            y = y.cuda()

        loss, cost = train_batch(model.net, model.criterion, model.optimizer, x, y)
        losses += [loss]
        costs += [cost]
        seq_lengths += [y.size(0)]
        repeats += [(x.size(0) - 2) / y.size(0)]
        # Update the progress bar
        progress_bar(batch_num, report_interval, loss)

        # Report
        if batch_num % report_interval == 0:
            mean_loss = np.array(losses[-report_interval:]).mean()
            mean_cost = np.array(costs[-report_interval:]).mean()
            mean_time = int(((get_ms() - start_ms) / report_interval) / batch_size)
            progress_clean()
            LOGGER.info(
                "Batch {} Loss: {} Cost: {} Time: {} ms/sequence".format(batch_num, mean_loss, mean_cost, mean_time))
            start_ms = get_ms()

        # Checkpoint
        if (checkpoint_interval != 0) and (batch_num % checkpoint_interval == 0):
            save_checkpoint(model.net, model.params.name, batch_num, losses, costs, seq_lengths, repeats,
                            checkpoint_path, seed)
