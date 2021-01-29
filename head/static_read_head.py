from head.static_head import *


class StaticReadHead(StaticHead):
    def __init__(self, args):
        super(StaticReadHead, self).__init__(args)

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.vector = [self.M, 1, 1, 3, 1]
        self.read_vector = self.vector * self.num_read_heads
        self.fc_read = nn.Linear(self.ctrl_size, sum(self.vector))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        w_perv = torch.zeros(batch_size, self.N)
        if self.is_cuda:
            w_perv = w_perv.cuda()
        return w_perv

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, x, last_w):
        out = self.fc_read(x)
        size = len(self.vector)
        segment = self.id * size
        K, B, G, S, L = split_cols(out, self.read_vector[segment: segment + size])
        w = self.address(K, B, G, S, L, last_w)
        r = self.memory.read(w)
        return r, w
