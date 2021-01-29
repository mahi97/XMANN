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
            std_dev = 1 / np.sqrt(self.N + self.M)
            nn.init.uniform_(self.mem_bias, -std_dev, std_dev)

        self.memory = None
        self.prev_mem = None

        # reset internal states
        self.usage = torch.zeros(self.batch_size, self.mem_hei)
        self.link = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei, self.mem_hei)
        self.preced = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei)

    def reset(self):
        """Initialize memory from bias, for start-of-sequence."""
        self.memory = self.mem_bias.clone().repeat(self.batch_size, 1, 1)

    def read(self, address, free_gate=None):
        """
        :param free_gate:
        :param address: Batched Tensor with Size of batch_size * N, contain value between 0 and 1 with sum equals to 1
        :return: Torch batched tensor with Size of batch_size * M, produce by sum over weighted elements of Memory
        """
        content = address.unsqueeze(1).matmul(self.memory).squeeze(1)

        self.usage = self._update_read_usage(address, free_gate)

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
        self._update_link(address)
        self._update_precedence_weights()

    def _update_read_usage(self, address, free_gate):
        """
        calculates the new usage after reading and freeing from memory
        variables needed:
            usage_vb: [batch_size x mem_hei]
            free_gate:  [batch_size x 1]
            address:    [batch_size x N]
        returns:
            usage_vb:      [batch_size x N]
        """
        free_read_weights_vb = free_gate.expand_as(address) * address
        psi = torch.prod(1. - free_read_weights_vb, 1)
        return self.usage * psi

    def _update_write_usage(self, address):
        """
        calculates the new usage after writing to memory
        variables needed:
            prev_usage_vb: [batch_size x mem_hei]
            address:    [batch_size x mem_hei]
        returns:
            usage_vb:      [batch_size x mem_hei]
        """
        # calculate the aggregated effect of all write heads
        # NOTE: how multiple write heads are delt w/ is not discussed in the paper
        # NOTE: this part is only shown in the source code
        write_weights = 1. - torch.prod(1. - address, 1)
        return self.usage + (1. - self.usage) * write_weights

    def _update_link(self, address):
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
        write_weights_i_vb = address.unsqueeze(3).expand_as(self.link)
        write_weights_j_vb = address.unsqueeze(2).expand_as(self.link)
        prev_preced_j_vb = self.preced.unsqueeze(2).expand_as(self.link)
        prev_link_scale_vb = 1 - write_weights_i_vb - write_weights_j_vb
        new_link_vb = write_weights_i_vb * prev_preced_j_vb
        link_vb = prev_link_scale_vb * self.link + new_link_vb
        # Return the link with the diagonal set to zero, to remove self-looping edges.
        # TODO: set diag as 0 if there's a specific method to do that w/ later releases
        diag_mask_vb = (1 - torch.eye(self.mem_hei).unsqueeze(0).unsqueeze(0).expand_as(link_vb)).type(self.dtype)
        link_vb = link_vb * diag_mask_vb
        return link_vb

    def _update_precedence_weights(self, address):
        """
        calculates the new precedence weights given the current write weights
        the precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the
        precedence weights unchanged, but with sum close to one will replace the
        precedence weights.
        variables needed:
            prev_preced_vb: [batch_size x num_write_heads x mem_hei]
            wl_curr_vb:     [batch_size x num_write_heads x mem_hei]
        returns:
            preced_vb:      [batch_size x num_write_heads x mem_hei]
        """
        # write_sum_vb = torch.sum(self.wl_curr_vb, 2)              # 0.1.12
        write_sum_vb = torch.sum(address, 2, keepdim=True)  # 0.2.0
        return (1 - write_sum_vb).expand_as(self.preced) * self.preced + address

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
            wl_prev_vb: [batch_size x mem_hei]
                     -> containing the previous read weights w_{t-1}^r.
            num_write_heads: number of writing heads
            num_read_heads: number of reading heads
            forward:    boolean
                     -> indicating whether to follow the "future" (True)
                     -> direction in the link graph or the "past" (False)
                     -> direction
        returns:
            directional_weights_vb: [batch_size x mem_hei]
        """
        if forward:
            directional_weights = torch.bmm(last_address, self.link_vb.view(-1, self.N, self.N).transpose(1, 2))
        else:
            directional_weights = torch.bmm(last_address, self.link_vb.view(-1, self.N, self.N))
        return directional_weights.transpose(1, 2)

    def _allocation(self, epsilon=1e-6):
        """
        computes allocation by sorting usage, a = a_t[\phi_t[j]]
        variables needed:
            usage_vb: [batch_size x mem_hei]
                   -> indicating current memory usage, this is equal to u_t in
                      the paper when we only have one write head, but for
                      multiple write heads, one should update the usage while
                      iterating through the write heads to take into account the
                      allocation returned by this function
        returns:
            alloc_vb: [batch_size x num_write_heads x mem_hei]
        """
        # ensure values are not too small prior to cumprod
        usage = epsilon + (1 - epsilon) * self.usage
        # NOTE: we sort usage in ascending order
        sorted_usage_vb, indices_vb = torch.topk(usage, k=self.N, dim=1, largest=False)
        cat_sorted_usage_vb = torch.cat((torch.ones(self.batch_size, 1), sorted_usage_vb), 1)[:, :-1]
        prod_sorted_usage_vb = torch.cumprod(cat_sorted_usage_vb, dim=1)  # TODO: use this once the PR is ready
        # alloc_weight_vb = (1 - sorted_usage_vb) * prod_sorted_usage_vb  # equ. (1)            # 0.1.12
        alloc_weight_vb = (1 - sorted_usage_vb) * prod_sorted_usage_vb.squeeze()  # equ. (1)    # 0.2.0
        _, indices_vb = torch.topk(indices_vb, k=self.N, dim=1, largest=False)
        alloc_weight_vb = alloc_weight_vb.gather(1, indices_vb)
        return alloc_weight_vb

    def location_focus(self, alloc_gate, write_gate, wc):
        """
        Calculates freeness-based locations for writing to.
        This finds unused memory by ranking the memory locations by usage, for
        each write head. (For more than one write head, we use a "simulated new
        usage" which takes into account the fact that the previous write head
        will increase the usage in that area of the memory.)
        variables needed:
            usage_vb:         [batch_size x mem_hei]
                           -> representing current memory usage
            write_gate_vb:    [batch_size x num_write_heads x 1]
                           -> /in [0, 1] indicating how much each write head
                              does writing based on the address returned here
                              (and hence how much usage increases)
        returns:
            alloc_weights_vb: [batch_size x num_write_heads x mem_hei]
                            -> containing the freeness-based write locations
                               Note that this isn't scaled by `write_gate`;
                               this scaling must be applied externally.
        """
        alloc_weight = self._allocation()
        # update usage to take into account writing to this new allocation
        # NOTE: checked: if not operate directly on _vb.data, then the _vb
        # NOTE: outside of this func will not change
        self.usage += (1 - self.usage) * self.write_gate[:, :].expand_as(self.usage) * alloc_weight
        # pack the allocation weights for write heads into one tensor
        w = self.write_gate.expand_as(alloc_weight) * (self.alloc_gate.expand_as(self.wc_vb) * alloc_weight
                                                       + (1. - self.alloc_gate_vb.expand_as(self.wc_vb)) * self.wc_vb)
        return w
