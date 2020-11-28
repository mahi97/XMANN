import torch
import torch.nn as nn

import memory


class NTM(nn.Module):

    def __init__(self, num_inputs, num_outputs, controller_size, controller_layers, num_read_heads, num_write_heads, N,
                 M):
        """Initialize an NTM.
        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(NTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.N = N
        self.M = M

        # Create the NTM components
        memory = Memory(N, M)
        controller = LSTMController(num_inputs + M * (num_read_heads), controller_size, controller_layers)
        # controller = FFController(num_inputs + M*num_heads, controller_size, controller_layers)
        heads = nn.ModuleList([ReadHead(memory, controller) for _ in range(num_read_heads)])
        heads += [WriteHead(memory, controller) for _ in range(num_write_heads)]

        self.data_path = DataPath(num_inputs, num_outputs, controller, memory, heads)
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.data_path.create_new_state(batch_size)

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs)
        if CUDA:
            x = x.cuda()
        o, self.previous_state = self.data_path(x, self.previous_state)
        return o, self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)

        return num_params
