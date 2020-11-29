"""Copy Task NTM model."""

from attr import attrs, attrib, Factory
import random
import numpy as np

import torch
from torch import nn
from torch import optim

from model import Model
from model import ModelParams


class CopyTask(object):
    def __init__(self):
        self.model = CopyTaskModel
        self.param = CopyTaskParams


def data_loader(num_batches, batch_size, seq_width, min_len, max_len, is_cuda=False):
    """Generator of random sequences for the copy task.
    Creates random batches of "bits" sequences.
    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]
    :param is_cuda: Generating data in GPU Memory
    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length.
    :param max_len: Sequence maximum length.
    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    for batch_num in range(num_batches):

        # All batches have the same sequence length
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0  # delimiter in our control channel
        outp = seq.clone()
        if is_cuda:
            inp = inp.cuda()
            outp = outp.cuda()

        yield batch_num + 1, inp.float(), outp.float()


@attrs
class CopyTaskParams(object):
    name = attrib(default="copy-task")
    memory = attrib(default='static')
    memory_init = attrib(default='const')
    controller = attrib(default='LSTM')
    data_path = attrib(default='NTM')
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)
    num_read_heads = attrib(default=1, converter=int)
    num_write_heads = attrib(default=1, converter=int)
    sequence_width = attrib(default=8, converter=int)
    sequence_min_len = attrib(default=1, converter=int)
    sequence_max_len = attrib(default=20, converter=int)
    memory_n = attrib(default=128, converter=int)
    memory_m = attrib(default=20, converter=int)
    num_batches = attrib(default=50000, converter=int)
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=1e-4, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)
    is_cuda = attrib(default=False, converter=bool)


@attrs
class CopyTaskModel(object):
    params = attrib(default=Factory(CopyTaskParams))
    net = attrib()
    data_loader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        model_params = ModelParams(
            memory=self.params.memory,
            controller=self.params.controller,
            data_path=self.params.data_path,
            num_inputs=self.params.sequence_width + 1,
            num_outputs=self.params.sequence_width,
            num_hidden=self.params.controller_layers,
            num_layers=self.params.controller_layers,
            controller_size=self.params.controller_size,
            num_read_heads=self.params.num_read_heads,
            num_write_heads=self.params.num_write_heads,
            memory_size=self.params.memory_n,
            word_size=self.params.memory_m,
            memory_init=self.params.memory_init,
            batch_size=self.params.batch_size,
            is_cuda=self.params.is_cuda
        )
        net = Model(model_params)
        if self.params.is_cuda:
            net = net.cuda()
        return net

    @data_loader.default
    def default_dataloader(self):
        return data_loader(self.params.num_batches, self.params.batch_size,
                           self.params.sequence_width,
                           self.params.sequence_min_len, self.params.sequence_max_len)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
