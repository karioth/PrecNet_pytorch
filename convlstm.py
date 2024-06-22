# @title ConvLSTM
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, norm = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding='same')
        self.norm = norm
        if self.norm:
            self.norm_in = nn.InstanceNorm2d(hidden_size, affine=False, track_running_stats=True)
            self.norm_remember = nn.InstanceNorm2d(hidden_size, affine=False, track_running_stats=True)
            self.norm_out = nn.InstanceNorm2d(hidden_size, affine=False, track_running_stats=True)
            self.norm_cell = nn.InstanceNorm2d(hidden_size, affine=False, track_running_stats=True)
            self.norm_cy = nn.InstanceNorm2d(hidden_size, affine=False, track_running_stats=True)

    def forward(self, input_, prev_state):
        prev_hidden, prev_cell = prev_state

        if prev_hidden.is_cuda == False  and torch.cuda.is_available():
           prev_hidden = prev_hidden.cuda()

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        if self.norm: 
            # Apply instance normalization
            in_gate = self.norm_in(in_gate)
            remember_gate = self.norm_remember(remember_gate)
            out_gate = self.norm_out(out_gate)
            cell_gate = self.norm_cell(cell_gate)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        if prev_cell.is_cuda == False and torch.cuda.is_available():
           prev_cell = prev_cell.cuda()

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        
        if self.norm:
            cell = self.norm_cy(cell)

        hidden = out_gate * f.tanh(cell)

        return hidden, cell
