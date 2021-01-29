from head.base_head import *


class StaticHead(BaseHead):
    def __init__(self, args):
        super(StaticHead, self).__init__(args)

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def content_address(self, key_vector, key_strength):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param key_vector: The key vector.
        :param key_strength: The key strength (focus).
        """
        wc = F.softmax(key_strength * F.cosine_similarity(key_vector.unsqueeze(1), self.memory.memory, dim=2), dim=1)
        return wc

    def location_address(self, wc, gate, shift, sharpen, last_address):
        wg = (gate * wc) + (1 - gate) * last_address
        ws = self.convolve(wg, shift, self.batch_size)
        ws = (ws ** sharpen)
        wt = torch.true_divide(ws, torch.sum(ws, dim=1).view(-1, 1) + 1e-16)

        return wt

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
        # Handle Activations
        k = torch.tanh(key_vector)
        B = F.softplus(key_strength)
        g = torch.sigmoid(gate)
        s = F.softmax(shift, dim=1)
        L = 1 + F.softplus(sharpen)

        wc = self.content_address(k, B)
        wt = self.location_address(wc, g, s, L, last_address)

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
