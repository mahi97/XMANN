import torch
import torch.nn as nn

from controller.controllers import *
from data_path.data_paths import *
from memory.memories import *
from head.heads import *

from head.base_head import HeadsParams
from memory.base_memory import MemoryParams
from controller.base_controller import ControllerParams

from data_path.base_data_path import *

from attr import attrs, attrib, Factory


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.num_inputs = args.num_inputs
        self.num_outputs = args.num_outputs
        self.controller_size = args.controller_size
        self.num_layers = args.num_layers
        self.num_read_heads = args.num_read_heads
        self.num_write_heads = args.num_write_heads
        self.N = args.memory_size
        self.M = args.word_size
        self.is_cuda = args.is_cuda
        self.batch_size = args.batch_size

        memory_param = MemoryParams(memory_size=args.memory_size,
                                    word_size=args.word_size,
                                    init_mode=args.memory_init,
                                    batch_size=args.batch_size,
                                    is_cuda=args.is_cuda)
        self.memory = MEMORIES[args.memory](memory_param)

        controller_params = ControllerParams(num_inputs=args.num_inputs + (args.word_size * args.num_read_heads),
                                             num_outputs=args.controller_size,
                                             num_hidden=args.num_hidden,
                                             num_layers=args.num_layers)
        controller = CONTROLLERS[args.controller](controller_params)

        head_params = HeadsParams(controller=controller, memory=self.memory, is_cuda=args.is_cuda)
        heads = nn.ModuleList([HEADS[args.read_head](head_params) for _ in range(args.num_read_heads)])
        heads += [HEADS[args.write_head](head_params) for _ in range(args.num_write_heads)]

        data_path_params = DataPathParams(num_inputs=self.num_inputs,
                                          num_outputs=self.num_outputs,
                                          controller=controller,
                                          memory=self.memory,
                                          heads=heads,
                                          is_cuda=args.is_cuda)
        self.data_path = DATA_PATHS[args.data_path](data_path_params)

        self.previous_state = None

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

    def init_sequence(self):
        """Initializing the state."""
        self.memory.reset()
        self.previous_state = self.data_path.create_new_state(self.batch_size)

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs)
        if self.is_cuda:
            x = x.cuda()
        o, self.previous_state = self.data_path(x, self.previous_state)
        return o, self.previous_state


@attrs
class ModelParams(object):
    memory = attrib(default='static')
    controller = attrib(default='LSTM')
    data_path = attrib(default='NTM')
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
