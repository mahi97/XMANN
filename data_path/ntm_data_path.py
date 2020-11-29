from data_path.base_data_path import *

import torch
import torch.nn as nn


class NTM(BaseDataPath):

    def __init__(self, args):
        super(NTM, self).__init__(args)

    def forward(self, x, prev_state):
        """DataPath forward function.
        :param x: input vector (batch_size x num_inputs)
        :param prev_state: The previous state of the DataPath
        """
        # Unpack the previous state
        prev_reads, prev_controller_state, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        if self.is_cuda:
            x = x.cuda()
            for i in range(len(prev_reads)):
                prev_reads[i] = prev_reads[i].cuda()

        inp = torch.cat([x] + prev_reads, dim=1)
        if self.is_cuda:
            inp = inp.cuda()
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                r, head_state = head(controller_outp, prev_head_state)
                if self.is_cuda:
                    r = r.cuda()
                    head_state = head_state.cuda()
                reads += [r]
            else:
                head_state = head(controller_outp, prev_head_state)
                if self.is_cuda:
                    head_state = head_state.cuda()
            heads_states += [head_state]

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = torch.sigmoid(self.fc(inp2))
        # o = torch.sigmoid(self.fc(controller_outp))

        # Pack the current state
        state = (reads, controller_state, heads_states)

        return o, state
