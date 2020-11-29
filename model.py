import torch
import torch.nn as nn

from controller.controllers import *
from memory.memories import *
from head.heads import *

from head.base_head import HeadsParams
from memory.base_memory import MemoryParams
from controller.base_controller import ControllerParams


from attr import attrs, attrib, Factory


class BaseNetwork(nn.Module):

    def __init__(self, args):
        """Initialize an NTM.
        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(BaseNetwork, self).__init__()

        self.num_inputs = args.num_inputs
        self.num_outputs = args.num_outputs
        self.controller_size = args.controller_size
        self.num_layers = args.num_layers
        self.num_read_heads = args.num_read_heads
        self.num_write_heads = args.num_write_heads
        self.N = args.memory_size
        self.M = args.word_size

        memory_param = MemoryParams(memory_size=args.memory_size,
                                    word_size=args.word_size,
                                    init_mode=args.memory_init,
                                    batch_size=args.batch_size,
                                    is_cuda=args.is_cuda)
        self.memory = MEMORIES[args.memory](memory_param)

        controller_params = ControllerParams(args.num_inputs + args.word_size * args.num_read_heads,
                                             args.num_outputs,
                                             args.num_hidden,
                                             args.num_layers)
        self.controller = CONTROLLERS[args.controller](controller_params)

        head_params = HeadsParams(controller=self.controller, memory=self.memory, is_cuda=args.is_cuda)
        self.heads = nn.ModuleList([HEADS[args.read_head](head_params) for _ in range(args.num_read_heads)])
        self.heads += [HEADS[args.write_head](head_params) for _ in range(args.num_write_heads)]

        self.data_path = DataPath(num_inputs, num_outputs, controller, memory, heads)
    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)

        return num_params


@attrs
class NetworkParams(object):
    memory = attrib(default='static')
    controller = attrib(default='LSTM')
    num_inputs = attrib(default=-1, converter=int)
    num_outputs = attrib(default=-1, converter=int)
    num_hidden = attrib(default=-1, converter=int)
    num_layers = attrib(default=-1, converter=int)
    controller_size = attrib(default=-1, converter=int)
    read_head = attrib(default='static-read')
    write_head = attrib(default='static-write')
    num_read_heads = attrib(default=-1, converter=int)
    num_write_heads = attrib(default=-1, converter=int)
    memory_size = attrib(default=-1, converter=int)
    word_size = attrib(default=-1, converter=int)
    memory_init = attrib(default='const')
    batch_size = attrib(default=-1, converter=int)
    is_cuda = attrib(default=False, converter=bool)