from controller.base_controller import *
import torch.nn as nn


class FFController(BaseController):
    def __init__(self, num_inputs, num_outputs, num_hidden):
        super(FFController, self).__init__(num_inputs, num_outputs, num_hidden)

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.4)
        nn.init.normal_(self.fc1.bias, std=0.01)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.4)
        nn.init.normal_(self.fc2.bias, std=0.01)


    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out
