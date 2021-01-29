from head.static_head import *


class StaticWriteHead(StaticHead):
    def __init__(self, args):
        super(StaticWriteHead, self).__init__(args)

        # Corresponding to K, B, G, S, L, add, erase
        self.vector = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.write_vector = self.vector * self.num_write_heads
        self.fc_write = nn.Linear(self.ctrl_size, sum(self.vector))
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

    def forward(self, x, last_w):
        out = self.fc_write(x)
        size = len(self.vector)
        segment = self.id * size
        K, B, G, S, L, A, E = split_cols(out, self.write_vector[segment: segment + size])
        w = self.address(K, B, G, S, L, last_w)
        self.memory.write(w, torch.sigmoid(E), torch.tanh(A))
        return w
