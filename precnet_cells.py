import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from .convlstm import ConvLSTMCell

class ErrorCell(nn.Module):
  '''Single Error Cell'''
  def __init__(self):
    super(ErrorCell,self).__init__()

  def forward(self, prediction, target):
        if torch.cuda.is_available() and prediction.is_cuda == False:
           prediction = prediction.cuda()
        if torch.cuda.is_available() and target.is_cuda == False:
           target = target.cuda()
        errors = f.relu(torch.cat((target - prediction, prediction - target), 1))

        # mean_errors = errors.mean(dim=(0, 2, 3), keepdim=True)
        # variance_errors = ((errors - mean_errors) ** 2).mean(dim=(0, 2, 3), keepdim=True)

        # epsilon = 1e-8

        # precision = 1 / (variance_errors + epsilon)
        # weighted_errors = errors * precision
    
        return errors

class PredictionCell(nn.Module):
  '''Single PredictionCell'''
  def __init__(self, in_channels, hidden_size):
    super(PredictionCell,self).__init__()
    self.in_channels_up, self.in_channels_down = in_channels
    self.hidden_size = hidden_size

    if self.in_channels_up is not None:

        self.convlstm_up = ConvLSTMCell(self.in_channels_up, self.hidden_size)

    self.convlstm_down = ConvLSTMCell(self.in_channels_down, self.hidden_size)

  def forward(self, error, r_state, c_state, up=False):
        if torch.cuda.is_available() and error.is_cuda == False:
           error = error.cuda()

        if up:
          r_state, c_state = self.convlstm_up(error, (r_state, c_state))
        else:
          r_state, c_state = self.convlstm_down(error, (r_state, c_state))
        return r_state, c_state
