"""Associative Recall Task NTM model."""

from attr import attrs, attrib, Factory
import random
import numpy as np

import torch
from torch import nn
from torch import optim

from network.networks import NETWORKS


class AssociativeRecallTask:
    def __init__(self):
        self.model = AssociativeRecallTaskModel
        self.param = AssociativeRecallTaskParams


# Generator of randomized test sequences
def data_loader(num_batches, batch_size, seq_width, seq_len, repeat_min, repeat_max):
    """Generator of random sequences for the Associative Recall task.
    Creates random batches of "bits" sequences.
    All the sequences within each batch have the same length.
    The length is between `min_len` to `max_len`
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

    for batch_num in range(num_batches):

        # All batches have the same sequence length and number of reps
        reps = np.random.randint(repeat_min, repeat_max + 1)

        # Generate the sequence
        seq = np.random.binomial(1, 0.5, (reps, seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        # The input includes 2 additional channels input and query delimeter
        # The input includes 1 additional seq for start and 1 additional repeat for query
        # and 1 additional seq for end-of-sequence
        # and 3(seq_len) additional empty for answer
        inp = torch.zeros((seq_len + 1) * (reps + 1) + 1, batch_size, seq_width + 2)
        for r in range(reps):
            inp[r * (seq_len + 1), :, seq_width] = 1.0  # StartPos
            inp[r * (seq_len + 1) + 1:(r + 1) * (seq_len + 1), :, :seq_width] = seq[r].clone()  # Sequence

        query_index = random.randint(0, reps - 2)
        query = seq[query_index].clone()
        answer = seq[query_index + 1].clone()

        inp[(reps) * (seq_len + 1), :, seq_width + 1] = 1.0  # Start Query
        inp[(reps) * (seq_len + 1) + 1:(reps + 1) * (seq_len + 1), :, :seq_width] = query.clone()  # Query
        inp[(reps + 1) * (seq_len + 1), :, seq_width + 1] = 1.0  # End Query

        # Left the rest empty

        # The output contain the repeated sequence + end marker
        outp = torch.zeros(seq_len, batch_size, seq_width + 2)
        outp[:, :, :seq_width] = answer.clone()

        if CUDA:
            inp = inp.cuda()
            outp = outp.cuda()

        yield batch_num + 1, inp.float(), outp.float()


@attrs
class AssociativeRecallTaskParams(object):
    name = attrib(default="recall-task")
    network = attrib(default='NTM', converter=str)
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)
    num_read_heads = attrib(default=1, converter=int)
    num_write_heads = attrib(default=1, converter=int)
    sequence_width = attrib(default=6, converter=int)
    sequence_len = attrib(default=3, converter=int)
    repeat_min = attrib(default=2, converter=int)
    repeat_max = attrib(default=6, converter=int)
    memory_n = attrib(default=128, converter=int)
    memory_m = attrib(default=20, converter=int)
    num_batches = attrib(default=50000, converter=int)
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=1e-4, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)
    is_cuda = attrib(default=False, converter=bool)


@attrs
class AssociativeRecallTaskModel(object):
    params = attrib(default=Factory(AssociativeRecallTaskParams))
    net = attrib()
    data_loader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # See dataloader documentation
        net = NETWORKS[self.params.network](self.params.sequence_width + 2, self.params.sequence_width + 2,
                  self.params.controller_size, self.params.controller_layers,
                  self.params.num_read_heads, self.params.num_write_heads,
                  self.params.memory_n, self.params.memory_m)
        if self.params.is_cuda:
            net = net.cuda()
        return net

    @data_loader.default
    def default_data_loader(self):
        return data_loader(self.params.num_batches,
                           self.params.batch_size,
                           self.params.sequence_width, self.params.sequence_len,
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
