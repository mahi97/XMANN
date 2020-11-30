from memory.base_memory import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DynamicMemory(BaseMemory):
    def __init__(self, args):
        super(DynamicMemory, self).__init__(args)

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(self.N, self.M))
        if self.init_mode == 'const':
            nn.init.constant_(self.mem_bias, 1e-6)
        elif self.init_mode == 'random':
            std_dev = 1 / np.sqrt(N + M)
            nn.init.uniform_(self.mem_bias, -std_dev, std_dev)

    def reset(self):
        pass

    def read(self, address):
        pass

    def write(self, address, erase_vector, add_vector):
        pass

    def address(self, key_vector, key_strength, gate, shift, sharpen, last_address):
        pass

    def convolve(self, wg, sg, batch_size):
        pass
