from memory.base_memory import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class StaticMemory(BaseMemory):
    def __init__(self, args):
        super(StaticMemory, self).__init__(args)

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(self.N, self.M))
        if self.init_mode == 'const':
            nn.init.constant_(self.mem_bias, 1e-6)
        elif self.init_mode == 'random':
            std_dev = 1 / np.sqrt(self.N + self.M)
            nn.init.uniform_(self.mem_bias, -std_dev, std_dev)

        self.memory = None
        self.prev_mem = None

    def reset(self):
        """Initialize memory from bias, for start-of-sequence."""
        self.memory = self.mem_bias.clone().repeat(self.batch_size, 1, 1)

    def read(self, address, free_gate=None):
        """
        :param free_gate: Nothing
        :param address: Batched Tensor with Size of batch_size * N, contain value between 0 and 1 with sum equals to 1
        :return: Torch batched tensor with Size of batch_size * M, produce by sum over weighted elements of Memory
        """
        return address.unsqueeze(1).matmul(self.memory).squeeze(1)

    def write(self, address, erase_vector, add_vector):
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        if self.is_cuda:
            self.memory = self.memory.cuda()
        erase = torch.matmul(address.unsqueeze(-1), erase_vector.unsqueeze(1))
        add = torch.matmul(address.unsqueeze(-1), add_vector.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add
