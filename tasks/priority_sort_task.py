"""Priority Sort Task"""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from model import *


class PrioritySortTask:
    def __init__(self):
        self.model = PrioritySortTaskModel
        self.param = PrioritySortTaskParams


def data_loader(num_batches,
                batch_size,
                seq_len,
                seq_width):
    """
    Data loader for the Priority Sort task.
    It generates set of binary sequences with
    attached a priority drawn from a given
    distribution.
    :param num_batches:
    :param batch_size:
    :param seq_width:
    :return:
    """
    for batch_num in range(num_batches):
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)

        # Add priority number (just a single one drawn from the uniform distribution)
        priority = np.random.uniform(-1, 1, (seq_len, batch_size, 1))
        priority = torch.from_numpy(priority)

        # Construct the input vectors
        inp[:seq_len, :, :seq_width] = seq
        inp[:seq_len, :, (seq_width):] = priority  # priority

        # Add the delimiter
        inp_delim = torch.zeros(seq_len + 1, batch_size, seq_width + 2)
        inp_delim[:(seq_len + 1), :, :(seq_width + 1)] = inp
        inp_delim[seq_len, :, seq_width + 1] = 1.0  # delimiter in our control channel

        # Construct the output which will be a sorterd version of
        # the sequences given by looking at the priority
        outp = inp_delim.numpy()

        # Strip all the binary vectors into a list
        # and sort the list by looking at the last column
        # (which will contain the priority)
        temp = []
        for i in range(len(outp)):
            temp.append(outp[i][0])
        del temp[-1]  # Remove the delimiter
        temp.sort(key=lambda x: x[seq_width], reverse=True)  # Sort elements descending order

        # Keep only the highest entries as specified in the paper.
        # This means that for 20 entries we want to predict only the highest 16.
        # This will be done only if a sequence is larger than 4 elements.
        if len(temp) > 4:
            del temp[-4:]

        # FIXME
        # Ugly hack to present the tensor structure as the one
        # required by the framework
        layer = []
        for i in range(len(temp)):
            tmp_layer = []
            tmp_layer.append(np.array(temp[i]))
            layer.append(tmp_layer)

        # Convert everything to numpy and to a tensor
        outp = torch.from_numpy(np.array(layer))

        yield batch_num + 1, inp_delim.float(), outp.float()


@attrs
class PrioritySortTaskParams(object):
    name = attrib(default="priority-sort-task")
    memory = attrib(default='static')
    memory_init = attrib(default='random')
    controller = attrib(default='LSTM')
    data_path = attrib(default='NTM')
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=2, converter=int)
    num_read_heads = attrib(default=5, converter=int)
    num_write_heads = attrib(default=5, converter=int)
    sequence_width = attrib(default=8, converter=int)
    sequence_min_len = attrib(default=1, converter=int)
    sequence_max_len = attrib(default=20, converter=int)
    memory_n = attrib(default=128, converter=int)
    memory_m = attrib(default=20, converter=int)
    num_batches = attrib(default=50000, converter=int)
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=3e-5, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)
    is_cuda = attrib(default=False, converter=bool)


@attrs
class PrioritySortTaskModel(object):
    params = attrib(default=Factory(PrioritySortTaskParams))
    cuda = attrib(default=False)
    net = attrib()
    data_loader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        model_params = ModelParams(
            memory=self.params.memory,
            controller=self.params.controller,
            data_path=self.params.data_path,
            num_inputs=self.params.sequence_width + 2,
            num_outputs=self.params.sequence_width + 2,
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
                           self.params.sequence_max_len, self.params.sequence_width)

    @criterion.default
    def default_criterion(self):
        criterion = nn.BCELoss()
        if self.cuda:
            criterion = criterion.cuda()

        return criterion

    @optimizer.default
    def default_optimizer(self):
        optimizer = optim.RMSprop(self.net.parameters(),
                                  momentum=self.params.rmsprop_momentum,
                                  alpha=self.params.rmsprop_alpha,
                                  lr=self.params.rmsprop_lr)

        return optimizer
