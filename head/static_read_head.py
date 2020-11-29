from head.base_head import *


class StaticReadHead(BaseHead):
    def __init__(self, memory, controller, is_cuda):
        super(StaticReadHead, self).__init__(memory, controller, is_cuda)

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_vector = [self.M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(self.ctrl_size, sum(self.read_vector))
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

    def forward(self, input, last_w):
        out = self.fc_read(input)
        K, B, G, S, L = split_cols(out, self.read_vector)
        w = self._address_memory(K, B, G, S, L, last_w)
        r = self.memory.read(w)
        return r, w