"""Repeated Copy Task NTM model."""

from attr import attrs, attrib, Factory
import random
import numpy as np

import torch
from torch import nn
from torch import optim

from network.networks import NETWORKS

class RepeatCopyTask:
    def __init__(self):
        self.model = RepeatCopyTaskModel
        self.param = RepeatCopyTaskParams


# Generator of randomized test sequences
def data_loader(num_batches, batch_size, seq_width, seq_min_len, seq_max_len, repeat_min, repeat_max, is_cuda=False):
    """Generator of random sequences for the repeat copy task.
    Creates random batches of "bits" sequences.
    All the sequences within each batch have the same length.
    The length is between `min_len` to `max_len`
    :param is_cuda: Use
    :param num_batches: Total number of batches to generate.
    :param batch_size: Batch size.
    :param seq_width: The width of each item in the sequence.
    :param seq_min_len: Sequence minimum length.
    :param seq_max_len: Sequence maximum length.
    :param repeat_min: Minimum repeatitions.
    :param repeat_max: Maximum repeatitions.
    NOTE: The input width is `seq_width + 2`. One additional input
    is used for the delimiter, and one for the number of repetitions.
    The output width is `seq_width` + 1, the additional input is used
    by the network to generate an end-marker, so we can be sure the
    network counted correctly.
    """
    # Some normalization constants
    reps_mean = (repeat_max + repeat_min) / 2
    reps_var = (((repeat_max - repeat_min + 1) ** 2) - 1) / 12
    reps_std = np.sqrt(reps_var) + 1e-16

    def rpt_normalize(reps):
        return (reps - reps_mean) / reps_std

    for batch_num in range(num_batches):

        # All batches have the same sequence length and number of reps
        seq_len = random.randint(seq_min_len, seq_max_len)
        reps = random.randint(repeat_min, repeat_max)

        # Generate the sequence
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        # The input includes 2 additional channels, for end-of-sequence and num-reps
        inp = torch.zeros(seq_len + 2, batch_size, seq_width + 2)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0
        inp[seq_len + 1, :, seq_width + 1] = rpt_normalize(reps)

        # The output contain the repeated sequence + end marker
        outp = torch.zeros(seq_len * reps + 1, batch_size, seq_width + 1)
        outp[:seq_len * reps, :, :seq_width] = seq.clone().repeat(reps, 1, 1)
        outp[seq_len * reps, :, seq_width] = 1.0  # End marker

        if is_cuda:
            inp = inp.cuda()
            outp = outp.cuda()

        yield batch_num + 1, inp.float(), outp.float()


@attrs
class RepeatCopyTaskParams(object):
    name = attrib(default="recopy-task")
    network = attrib(default='NTM', converter=str)
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)
    num_read_heads = attrib(default=1, converter=int)
    num_write_heads = attrib(default=1, converter=int)
    sequence_width = attrib(default=8, converter=int)
    sequence_min_len = attrib(default=1, converter=int)
    sequence_max_len = attrib(default=10, converter=int)
    repeat_min = attrib(default=1, converter=int)
    repeat_max = attrib(default=10, converter=int)
    memory_n = attrib(default=128, converter=int)
    memory_m = attrib(default=20, converter=int)
    num_batches = attrib(default=250000, converter=int)
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=1e-4, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)
    is_cuda = attrib(default=False, converter=bool)


@attrs
class RepeatCopyTaskModel(object):
    params = attrib(default=Factory(RepeatCopyTaskParams))
    net = attrib()
    data_loader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # See data_loader documentation
        net = NETWORKS[self.params.network](self.params.sequence_width + 2, self.params.sequence_width + 1,
                  self.params.controller_size, self.params.controller_layers,
                  self.params.num_read_heads, self.params.num_write_heads,
                  self.params.memory_n, self.params.memory_m)
        if self.params.is_cuda:
            net = net.cuda()
        return net

    @data_loader.default
    def default_data_loader(self):
        return data_loader(self.params.num_batches, self.params.batch_size,
                           self.params.sequence_width,
                           self.params.sequence_min_len, self.params.sequence_max_len,
                           self.params.repeat_min, self.params.repeat_max)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
