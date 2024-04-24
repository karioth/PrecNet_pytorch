import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from convlstmcell import ConvLSTMCell

class ErrorCell(nn.Module):
  '''Single Error Cell'''
  def __init__(self):
    super(ErrorCell,self).__init__()
    self.input_size = input_size

    def forward(self, prediction, target):
        error = f.relu(torch.cat((target - prediction, prediction - target), 1))
        return error

class PredictionCell(nn.Module):
  '''Single PredictionCell'''
  def __init__(self, input_size, hidden_size, error_init_size=None):
    '''
    Creates a Prediction unit: (error, top_down_state, r_state) -> r_state
    :param input_size: {'error': error_size, 'up_state': r_state_size}, r_state_size can be 0
    :param hidden_size: int, shooting dimensionality
    :param error_init_size: tuple, full size of initial (null) error
    '''
    super(PredictionCell,self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.error_init_size = error_init_size
    self.convlstm = ConvLSTMCell(input_size, hidden_size)

    def forward(self, error, r_state, c_state):
        if error is None:
            error = Variable(torch.zeros(self.error_init_size)) #We just started, initialize 0's error

        r_state, c_state = self.convlstm(error, (r_state, c_state))
        return r_state, c_state
