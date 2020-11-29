import torch.nn as nn
from attr import attrs, attrib, Factory


class BaseController(nn.Module):
    def __init__(self, args):
        super(BaseController, self).__init__()

        self.num_inputs = args.num_inputs
        self.num_hidden = args.num_hidden
        self.num_outputs = args.num_outputs
        self.num_layers = args.num_layers

    def reset_parameters(self):
        raise NotImplementedError('Not Implemented in base class')

    def size(self):
        return self.num_inputs, self.num_outputs

@attrs
class ControllerParams(object):
    num_inputs = attrib(default=1, converter=int)
    num_outputs = attrib(default=1, converter=int)
    num_hidden = attrib(default=1, converter=int)
    num_layers = attrib(default=1, converter=int)