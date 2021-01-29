from head.base_head import *


class DynamicHead(BaseHead):
    def __init__(self, args):
        super(DynamicHead, self).__init__(args)

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

    def address(self, key_vector, key_strength):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param key_vector: The key vector.
        :param key_strength: The key strength (focus).
        """
        # Handle Activations
        k = torch.tanh(key_vector)
        B = F.softplus(key_strength)
        wc = self.content_address(k, B)
        return wc
