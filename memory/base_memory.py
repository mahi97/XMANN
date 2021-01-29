import torch
import torch.nn as nn
from attr import attrs, attrib, Factory


class BaseMemory(nn.Module):
    def __init__(self, args):
        super(BaseMemory, self).__init__()

        self.N = args.memory_size
        self.M = args.word_size
        self.init_mode = args.init_mode
        self.batch_size = args.batch_size
        self.is_cuda = args.is_cuda
        self.num_write_heads = args.num_write_heads
        self.num_read_heads = args.num_read_heads

    def size(self):
        return self.N, self.M

    def reset(self):
        raise NotImplementedError('Not Implemented in base class')

    def read(self, address, free_gate=None):
        raise NotImplementedError('Not Implemented in base class')

    def write(self, address, erase_vector, add_vector):
        raise NotImplementedError('Not Implemented in base class')


@attrs
class MemoryParams(object):
    name = attrib(default='base-memory')
    memory_size = attrib(default=128, converter=int)
    word_size = attrib(default=8, converter=int)
    init_mode = attrib(default='const')
    batch_size = attrib(default=1, converter=int)
    is_cuda = attrib(default=False, converter=bool)
    num_read_heads = attrib(default=1, converter=int)
    num_write_heads = attrib(default=1, converter=int)