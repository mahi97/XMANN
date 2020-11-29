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

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, k, B, g, s, L, w_prev):
        # Handle Activations
        k = torch.tanh(k)
        B = F.softplus(B)
        g = torch.sigmoid(g)
        s = F.softmax(s, dim=1)
        L = 1 + F.softplus(L)

        w = self.memory.address(k, B, g, s, L, w_prev)

        return w


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
