import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attr import attrs, attrib, Factory

from head.base_head import BaseHead


class DynamicHead(BaseHead):
    def __init__(self, args):
        super(DynamicHead, self).__init__(args)

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
