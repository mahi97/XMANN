from head.dynamic_head import *


class DynamicWriteHead(DynamicHead):
    def __init__(self, args):
        super(DynamicWriteHead, self).__init__(args)

        self.num_heads = args.num_read_heads
        self.num_modes = args.num_read_modes

        #                    K, B, alloc_gate, write_gate, add,    erase
        self.write_vector = [self.M, 1, 1, 1, self.M, self.M]
        self.fc_write = nn.Linear(self.ctrl_size, sum(self.write_vector))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        w_perv = torch.zeros(batch_size, self.N)
        if self.is_cuda:
            w_perv = w_perv.cuda()
        return w_perv

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, input, last_w):
        out = self.fc_write(input)
        alloc_gate, write_gate, add, erase = split_cols(out, self.write_vector)
        allocation_weight = self._allocation(self.memory.usage_vb)
        self._location_focus(self.memory.usage_vb)
        w = self.address(K, B, G, S, L, last_w)

        self.memory.write(w, torch.sigmoid(E), torch.tanh(A))
        return w

    def _allocation(self, usage_vb, epsilon=1e-6):
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
        usage_vb = epsilon + (1 - epsilon) * usage_vb
        # NOTE: we sort usage in ascending order
        sorted_usage_vb, indices_vb = torch.topk(usage_vb, k=self.mem_hei, dim=1, largest=False)
        cat_sorted_usage_vb = torch.cat((torch.ones(self.batch_size, 1), sorted_usage_vb), 1)[:, :-1]
        prod_sorted_usage_vb = torch.cumprod(cat_sorted_usage_vb, dim=1)
        # alloc_weight_vb = (1 - sorted_usage_vb) * prod_sorted_usage_vb  # equ. (1)            # 0.1.12
        alloc_weight_vb = (1 - sorted_usage_vb) * prod_sorted_usage_vb.squeeze()  # equ. (1)    # 0.2.0
        _, indices_vb = torch.topk(indices_vb, k=self.mem_hei, dim=1, largest=False)
        alloc_weight_vb = alloc_weight_vb.gather(1, indices_vb)
        return alloc_weight_vb

    def _location_focus(self, usage_vb):
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
        alloc_weights_vb = []
        for i in range(self.num_heads):
            alloc_weights_vb.append(self._allocation(usage_vb))
            # update usage to take into account writing to this new allocation
            # NOTE: checked: if not operate directly on _vb.data, then the _vb
            # NOTE: outside of this func will not change
            usage_vb += (1 - usage_vb) * self.write_gate_vb[:, i, :].expand_as(usage_vb) * alloc_weights_vb[i]
        # pack the allocation weights for write heads into one tensor
        alloc_weight_vb = torch.stack(alloc_weights_vb, dim=1)
        self.wl_curr_vb = self.write_gate_vb.expand_as(alloc_weight_vb) * (self.alloc_gate_vb.expand_as(self.wc_vb) * alloc_weight_vb + \
                                                                           (1. - self.alloc_gate_vb.expand_as(self.wc_vb)) * self.wc_vb)
