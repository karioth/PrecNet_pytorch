import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from .precnet_cells import ErrorCell, PredictionCell

class PrecNetModel(nn.Module):
  def __init__(self, hidden_sizes, r_hidden_sizes):
        super(PrecNetModel,self).__init__()
        self.num_of_layers = len(r_hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.r_hidden_sizes = r_hidden_sizes

        error_units = []
        pred_units = []
        ahat_units = []
        
        for i in range(self.num_of_layers):
            if i == self.num_of_layers - 1:  # Top layer
                in_channels = (None, self.hidden_sizes[i] * 2)
            else:
                in_channels = (self.hidden_sizes[i] * 2, self.hidden_sizes[i+1] * 2)
              
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

  def train_step(self, seq, optimizer, loss_fn, layer_loss_weights):
      optimizer.zero_grad()
      seq = seq.permute(0, 1, 4, 2, 3)
      input = Variable(seq.cuda())
      time_steps = input.size(1)
      states = self.init_states(input)
      loss = 0.0
      for t in range(time_steps):
        A = input[:,t]
        A = A.type(torch.cuda.FloatTensor)
        states, layer_errors = self(A, states, training=True)
        if t > 0: #ignore the first time_step
            weighted_errors = torch.mm(layer_errors, layer_loss_weights)
            target = torch.zeros_like(weighted_errors)
            loss += loss_fn(weighted_errors, target)

      loss.backward()
      optimizer.step()

      return loss

  def test_step(self, seq, loss_fn, layer_loss_weights):
      seq = seq.permute(0, 1, 4, 2, 3)
      input = Variable(seq.cuda())
      time_steps = input.size(1)
      states = self.init_states(input)
      loss = 0.0
      for t in range(time_steps):
          A = input[:,t]
          A = A.type(torch.cuda.FloatTensor)
          states, layer_errors = self(A, states, training=True)
          if t > 0: #ignore the first time_step
              weighted_errors = torch.mm(layer_errors, layer_loss_weights)
              target = torch.zeros_like(weighted_errors)
              loss += loss_fn(weighted_errors, target)
      return loss

  def predict(self, seq):
      seq = seq.permute(0, 1, 4, 2, 3)
      input = Variable(seq.cuda())
      time_steps = input.size(1)
      states = self.init_states(input)
      predictions = []
      for t in range(time_steps):
          A = input[:,t]
          A = A.type(torch.cuda.FloatTensor)
          states, prediction = self(A, states, training=False)
          predictions.append(prediction)

      predictions = torch.stack(predictions, dim = 1)
      return predictions

