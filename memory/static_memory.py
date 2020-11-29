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
            std_dev = 1 / np.sqrt(N + M)
            nn.init.uniform_(self.mem_bias, -std_dev, std_dev)

    def reset(self):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = self.batch_size
        self.memory = self.mem_bias.clone().repeat(self.batch_size, 1, 1)

    def read(self, address):
        """
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

    def address(self, key_vector, key_strength, gate, shift, sharpen, last_address):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param key_vector: The key vector.
        :param key_strength: The key strength (focus).
        :param gate: Scalar interpolation gate (with previous weighting).
        :param shift: Shift weighting.
        :param sharpen: Sharpen weighting scalar.
        :param last_address: The weighting produced in the previous time step.
        """
        wc = F.softmax(key_strength * F.cosine_similarity(key_vector.unsqueeze(1), self.memory, dim=2), dim=1)
        wg = (gate * wc) + (1 - gate) * last_address
        ws = self.convolve(wg, shift, self.batch_size)
        ws = (ws ** sharpen)
        wt = torch.true_divide(ws, torch.sum(ws, dim=1).view(-1, 1) + 1e-16)

        return wt

    def convolve(self, wg, sg, batch_size):
        """Circular convolution implementation."""
        result = torch.zeros(wg.size())
        for i in range(batch_size):
            w = wg[i]
            s = sg[i]
            assert s.size(0) == 3
            t = torch.cat([w[-1:], w, w[:1]])
            result[i] = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
        if self.is_cuda:
            result = result.cuda()
        return result
