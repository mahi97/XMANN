import torch
import torch.nn as nn

from attr import attrs, attrib, Factory


class BaseDataPath(nn.Module):
    """Base DataPath."""

    def __init__(self, args):
        """Initialize the DataPath.
        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`LSTMController`
        :param memory: :class:`Memory`
        :param heads: list of :class:`ReadHead` or :class:`WriteHead`
        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(BaseDataPath, self).__init__()

        self.num_inputs = args.num_inputs
        self.num_outputs = args.num_outputs
        self.controller = args.controller
        self.memory = args.memory
        self.heads = args.heads
        self.is_cuda = args.is_cuda

        self.N, self.M = self.memory.size()
        _, self.controller_size = self.controller.size()

        # Initialize the initial previous read values to random biases
        self.num_read_heads = args.num_read_heads
        self.num_write_heads = args.num_write_heads
        self.init_r = []
        for i, head in enumerate(self.heads):
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(i), init_r_bias.data)
                self.init_r += [init_r_bias]
                # head.id = self.num_read_heads
                # self.num_read_heads += 1
            # else:
                # head.id = self.num_write_heads
                # self.num_write_heads += 1

        assert self.num_read_heads > 0, "heads list must contain at least a single read head"

        self.fc = nn.Linear(self.controller_size + self.num_read_heads * self.M, self.num_outputs)
        self.reset_parameters()

    def create_new_state(self, batch_size):
        self.memory.reset()
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]

        return init_r, controller_state, heads_state

    def reset_parameters(self):
        # Initialize the linear layer
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, x, prev_state):
        raise NotImplementedError('Not implemented in base class!')


@attrs
class DataPathParams(object):
    num_inputs = attrib(default=-1, converter=int)
    num_outputs = attrib(default=-1, converter=int)
    controller = attrib(default=None)
    memory = attrib(default=None)
    heads = attrib(default=None)
    is_cuda = attrib(default=False, converter=bool)
    num_read_heads = attrib(default=1, converter=int)
    num_write_heads = attrib(default=1, converter=int)