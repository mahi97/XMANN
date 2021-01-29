from memory.base_memory import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
            std_dev = 1 / np.sqrt(self.N + self.M)
            nn.init.uniform_(self.mem_bias, -std_dev, std_dev)

        self.memory = None
        self.prev_mem = None

        # reset internal states
        self.usage_ts = torch.zeros(self.batch_size, self.mem_hei)
        self.link_ts = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei, self.mem_hei)
        self.preced_ts = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei)

        # reset the usage (for dynamic allocation) & link (for temporal link)
        self.usage_vb = Variable(self.usage_ts).type(self.dtype)
        self.link_vb = Variable(self.link_ts).type(self.dtype)
        self.preced_vb = Variable(self.preced_ts).type(self.dtype)

    def reset(self):
        """Initialize memory from bias, for start-of-sequence."""
        self.memory = self.mem_bias.clone().repeat(self.batch_size, 1, 1)

        # reset the usage (for dynamic allocation) & link (for temporal link)
        self.usage_vb = Variable(self.usage_ts).type(self.dtype)
        self.link_vb = Variable(self.link_ts).type(self.dtype)
        self.preced_vb = Variable(self.preced_ts).type(self.dtype)

    def read(self, address, free_gate=None):
        """
        :param free_gate:
        :param address: Batched Tensor with Size of batch_size * N, contain value between 0 and 1 with sum equals to 1
        :return: Torch batched tensor with Size of batch_size * M, produce by sum over weighted elements of Memory
        """
        content = address.unsqueeze(1).matmul(self.memory).squeeze(1)

        self.usage_vb = self.update_read_usage(address, free_gate)

        return content

    def write(self, address, erase_vector, add_vector):
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        if self.is_cuda:
            self.memory = self.memory.cuda()
        erase = torch.matmul(address.unsqueeze(-1), erase_vector.unsqueeze(1))
        add = torch.matmul(address.unsqueeze(-1), add_vector.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

        self._update_write_usage(address)

    def update_read_usage(self, address, free_gate):
        """
        calculates the new usage after reading and freeing from memory
        variables needed:
            usage_vb: [batch_size x mem_hei]
            free_gate:  [batch_size x num_heads x 1]
            address:    [batch_size x num_heads x mem_hei]
        returns:
            usage_vb:      [batch_size x mem_hei]
        """
        free_read_weights_vb = self.free_gate_vb.expand_as(address) * address
        psi_vb = torch.prod(1. - free_read_weights_vb, 1)
        return self.usage_vb * psi_vb

    def _update_write_usage(self, address):
        """
        calculates the new usage after writing to memory
        variables needed:
            prev_usage_vb: [batch_size x mem_hei]
            wl_prev_vb:    [batch_size x num_write_heads x mem_hei]
        returns:
            usage_vb:      [batch_size x mem_hei]
        """
        # calculate the aggregated effect of all write heads
        # NOTE: how multiple write heads are delt w/ is not discussed in the paper
        # NOTE: this part is only shown in the source code
        write_weights_vb = 1. - torch.prod(1. - address, 1)
        return self.usage_vb + (1. - self.usage_vb) * write_weights_vb

    def _update_link(self, prev_link_vb, prev_preced_vb):
        """
        calculates the new link graphs
        For each write head, the link is a directed graph (represented by a
        matrix with entries in range [0, 1]) whose vertices are the memory
        locations, and an edge indicates temporal ordering of writes.
        variables needed:
            prev_link_vb:   [batch_size x num_heads x mem_hei x mem_wid]
                         -> {L_t-1}, previous link graphs
            prev_preced_vb: [batch_size x num_heads x mem_hei]
                         -> {p_t}, the previous aggregated precedence
                         -> weights for each write head
            wl_curr_vb:     [batch_size x num_heads x mem_hei]
                         -> location focus of {t}
        returns:
            link_vb:        [batch_size x num_heads x mem_hei x mem_hei]
                         -> {L_t}, current link graph
        """
        write_weights_i_vb = self.wl_curr_vb.unsqueeze(3).expand_as(prev_link_vb)
        write_weights_j_vb = self.wl_curr_vb.unsqueeze(2).expand_as(prev_link_vb)
        prev_preced_j_vb = prev_preced_vb.unsqueeze(2).expand_as(prev_link_vb)
        prev_link_scale_vb = 1 - write_weights_i_vb - write_weights_j_vb
        new_link_vb = write_weights_i_vb * prev_preced_j_vb
        link_vb = prev_link_scale_vb * prev_link_vb + new_link_vb
        # Return the link with the diagonal set to zero, to remove self-looping edges.
        # TODO: set diag as 0 if there's a specific method to do that w/ later releases
        diag_mask_vb = Variable(1 - torch.eye(self.mem_hei).unsqueeze(0).unsqueeze(0).expand_as(link_vb)).type(self.dtype)
        link_vb = link_vb * diag_mask_vb
        return link_vb

    def _temporal_link(self, prev_link_vb, prev_preced_vb):
        link_vb = self._update_link(prev_link_vb, prev_preced_vb)
        preced_vb = self._update_precedence_weights(prev_preced_vb)
        return link_vb, preced_vb

    def directional_read_weights(self, num_write_heads, num_read_heads, forward, last_address):
        """
        calculates the forward or the backward read weights
        for each read head (at a given address), there are `num_writes` link
        graphs to follow. thus this function computes a read address for each of
        the `num_reads * num_writes` pairs of read and write heads.
        we calculate the forward and backward directions for each pair of read
        and write heads; hence we need to tile the read weights and do a sort of
        "outer product" to get this.
        variables needed:
            link_vb:    [batch_size x num_read_heads x mem_hei x mem_hei]
                     -> {L_t}, current link graph
            wl_prev_vb: [batch_size x num_read_heads x mem_hei]
                     -> containing the previous read weights w_{t-1}^r.
            num_write_heads: number of writing heads
            num_read_heads: number of reading heads
            forward:    boolean
                     -> indicating whether to follow the "future" (True)
                     -> direction in the link graph or the "past" (False)
                     -> direction
        returns:
            directional_weights_vb: [batch_size x num_read_heads x num_write_heads x mem_hei]
        """
        expanded_read_weights_vb = last_address.unsqueeze(1).expand_as(
            torch.Tensor(self.batch_size, num_write_heads, num_read_heads, self.N)).contiguous()
        if forward:
            directional_weights_vb = torch.bmm(expanded_read_weights_vb.view(-1, num_read_heads, self.N),
                                               self.link_vb.view(-1, self.N, self.N).transpose(1, 2))
        else:
            directional_weights_vb = torch.bmm(expanded_read_weights_vb.view(-1, num_read_heads, self.N),
                                               self.link_vb.view(-1, self.N, self.N))
        return directional_weights_vb.view(-1, num_write_heads, num_read_heads, self.N).transpose(1, 2)