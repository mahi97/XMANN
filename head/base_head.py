import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attr import attrs, attrib, Factory


class BaseHead(nn.Module):
    def __init__(self, args):
        super(BaseHead, self).__init__()

        self.memory = args.memory
        _, self.ctrl_size = args.controller.size()
        self.M = self.memory.M
        self.N = self.memory.N
        self.is_cuda = args.is_cuda
        self.batch_size = args.batch_size
        self.id = args.id
        self.num_read_heads = args.num_read_heads
        self.num_write_heads = args.num_write_heads

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError


def split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


@attrs
class HeadsParams(object):
    controller = attrib(default=None)
    memory = attrib(default=None)
    is_cuda = attrib(default=False, converter=bool)
    batch_size = attrib(default=1, converter=int)
    num_read_heads = attrib(default=1, converter=int)
    num_write_heads = attrib(default=1, converter=int)
    id = attrib(default=0, converter=int)