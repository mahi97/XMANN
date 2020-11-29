import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseHead(nn.Module):
    def __init__(self, memory, controller, is_cuda):
        super(BaseHead, self).__init__()

        self.memory = memory
        _, self.ctrl_size = controller.size()
        self.M = memory.M
        self.N = memory.N
        self.is_cuda = is_cuda

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
        k = F.tanh(k)
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
