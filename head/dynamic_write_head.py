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

    def forward(self, x):
        out = self.fc_write(x)
        k, b, alloc_gate, write_gate, add, erase = split_cols(out, self.write_vector)
        w = self.address(k, b)
        wl = self.memory.location_focus(alloc_gate, write_gate, w)
        self.memory.write(wl, torch.sigmoid(erase), torch.tanh(add))
        return wl
