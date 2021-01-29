from head.dynamic_head import *


class DynamicReadHead(DynamicHead):
    def __init__(self, args):
        super(DynamicReadHead, self).__init__(args)

        self.num_heads = args.num_read_heads
        self.num_modes = args.num_read_modes
        # Corresponding to k, β, free_gate, read_mode sizes from the paper
        self.vector = [self.M, 1, 1, self.num_modes]
        self.read_vector = self.vector * self.num_read_heads
        self.fc_read = nn.Linear(self.ctrl_size, sum(self.vector))

        # functional components
        self.usage_vb = None  # for dynamic allocation, init in _reset
        self.link_vb = None  # for temporal link, init in _reset
        self.preced_vb = None  # for temporal link, init in _reset

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

    def content_address(self, key_vector, key_strength):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param key_vector: The key vector.
        :param key_strength: The key strength (focus).
        """
        wc = F.softmax(key_strength * F.cosine_similarity(key_vector.unsqueeze(1), self.memory.memory, dim=2), dim=1)
        return wc

    def location_address(self, wc, num_head, read_mode, last_address):
        forward_weights = self.memory.directional_read_weights(True, last_address)
        backward_weights = self.memory.directional_read_weights(False, last_address)
        backward_mode = read_mode[:, 0]
        forward_mode  = read_mode[:, 1]
        content_mode  = read_mode[:, 2]

        wl = content_mode.expand_as(wc) * wc
        wl += torch.sum(forward_mode.unsqueeze(1).expand_as(forward_weights) * forward_weights, -1)
        wl += torch.sum(backward_mode.unsqueeze(1).expand_as(backward_weights) * backward_weights, -1)

        return wl

    def address(self, key_vector, key_strength, num_head, read_mode, last_address):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param last_address:
        :param key_vector: The key vector.
        :param key_strength: The key strength (focus).
        :param read_mode:
        :param num_head:
        """
        # Handle Activations
        k = torch.tanh(key_vector)
        B = F.softplus(key_strength)

        wc = self.content_address(k, B)
        wl = self.location_address(wc, num_head, read_mode, last_address)

        return wl

    def forward(self, x, last_w):
        out = self.fc_read(x)
        size = len(self.vector)
        segment = self.id * size
        k, b, free_gate, read_modes = split_cols(out, self.read_vector[segment: segment + size])
        w = self.address(k, b, self.num_heads, read_modes, last_w)
        r = self.memory.read(w, F.sigmoid(free_gate))
        return r, w
