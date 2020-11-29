import torch
import numpy as np
from controller.base_controller import *


class LSTMController(BaseController):
    def __init__(self, args):
        super(LSTMController, self).__init__(args)

        self.lstm = nn.LSTM(self.num_inputs, self.num_outputs, self.num_layers)

        self.lstm_h = nn.Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c = nn.Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs + self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state
