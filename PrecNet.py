import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from convlstmcell import ConvLSTMCell

class ErrorCell(nn.Module):
  '''Single Error Cell'''
  def __init__(self):
    super(ErrorCell,self).__init__()

  def forward(self, prediction, target):
        if torch.cuda.is_available() and prediction.is_cuda == False:
           prediction = prediction.cuda()
        if torch.cuda.is_available() and target.is_cuda == False:
           target = target.cuda()
        error = f.relu(torch.cat((target - prediction, prediction - target), 1))
        return error

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



class PrecNetModel(nn.Module):
  def __init__(self, hidden_sizes, r_hidden_sizes):
        super(PrecNetModel,self).__init__()
        self.num_of_layers = len(r_hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.r_hidden_sizes = r_hidden_sizes

        error_units = []
        pred_units = []
        ahat_units = []
        for i, channels in enumerate([(6,120),(120,240),(None, 240)]):
            in_channels = channels

            pred_units.append(PredictionCell(in_channels, r_hidden_sizes[i]))
            ahat_units.append(nn.Conv2d(r_hidden_sizes[i], hidden_sizes[i], kernel_size=3, padding='same'))
            error_units.append(ErrorCell())

        self.error_units = nn.ModuleList(error_units)
        self.pred_units = nn.ModuleList(pred_units)
        self.ahat_units = nn.ModuleList(ahat_units)

  def forward(self, A, states, training = False):
        r_states, c_states, errors = states
        a_hats = [None] * self.num_of_layers
        #prediction phase
        for l in reversed(range(self.num_of_layers)):
            #get new R's, C's
            if l == self.num_of_layers - 1:
                r_state, c_state = self.pred_units[l](errors[l], r_states[l], c_states[l]) #convlstm down
            else:
                upsamp_error = f.interpolate(errors[l+1], scale_factor = 2)
                r_state, c_state = self.pred_units[l](upsamp_error, r_states[l], c_states[l])
            #get new Ahats's
            ahat = self.ahat_units[l](r_state)
            #use to calculate errors:
            if l == 0:
                ahat = torch.clamp(ahat, max = 1) #do satlu
                error = self.error_units[l](prediction=ahat, target=A)
                output = ahat # save the prediction for inference
            else:
                pool_r = f.max_pool2d(r_states[l-1], 2, 2)
                error = self.error_units[l](prediction=ahat, target = pool_r)
            #update everything
            a_hats[l] = ahat
            r_states[l] = r_state
            c_states[l] = c_state
            errors[l] = error

        #correction phase:
        for l in range(self.num_of_layers):
            if l == 0:
                pass
            else:
                pool_r = f.max_pool2d(r_states[l-1], 2, 2)
                error = self.error_units[l](prediction=a_hats[l], target = pool_r)
                errors[l] = error

            if l < self.num_of_layers - 1:
                r_state, c_state = self.pred_units[l](error, r_states[l], c_states[l], up=True) #convlstm up
                r_states[l] = r_state
                c_states[l] = c_state
          
            if training:
                layer_error = torch.mean(torch.flatten(errors[l], start_dim=1), dim=1, keepdim=True) 
                all_error = layer_error if l == 0 else torch.cat((all_error, layer_error), dim=1)
                output = all_error
        
        states = (r_states, c_states, errors)
        return states, output


  def init_states(self, input):
    r_states = [None] * self.num_of_layers
    c_states = [None] * self.num_of_layers
    errors = [None] * self.num_of_layers
    
    height = input.size(-2)
    width = input.size(-1)
    batch_size = input.size(0)

    for l in range(self.num_of_layers):

        errors[l] = Variable(torch.zeros(batch_size, 2*self.hidden_sizes[l], height, width))
        r_states[l]= Variable(torch.zeros(batch_size, self.r_hidden_sizes[l], height, width))
        c_states[l] = Variable(torch.zeros(batch_size, self.r_hidden_sizes[l], height, width))

        height = height // 2
        width = width // 2

    return (r_states, c_states, errors)
