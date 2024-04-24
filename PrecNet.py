class PrecNetModel(nn.module):
  def __init__(self, hidden_sizes, r_hidden_sizes, error_init_sizes):
        super(PrecNetModel,self).__init__()
        self.num_of_layers = len(r_hidden_sizes)
        error_units = []
        pred_units = []
        ahat_units = []
        for i in range(self.num_of_layers):
            if i == 0:
                in_channels = 2 * hidden_sizes[i]
            else:
                in_channels = 2 * hidden_sizes[i] + r_hidden_sizes[i-1]

            pred_units.append(PredictionCell(in_channels, r_hidden_sizes[i], error_init_size = error_init_sizes[i]))
            ahat_units.append(nn.Conv2d(r_hidden_sizes[i], hidden_sizes[i], kernel_size=3, padding='same'))
            error_units.append(ErrorCell())

        self.error_units = nn.ModuleList(error_units)
        self.pred_units = nn.ModuleList(pred_units)
        self.ahat_units = nn.ModuleList(ahat_units)

  def forward(self, input, states, training = False):
        r_states, c_states, errors = states
        a_hats = [None] * self.num_of_layers

        #prediction phase
        for l in reversed(range(self.num_of_layers)):
            #get new R's, C's
            if l == self.num_of_layers - 1:
                r_state, c_state = self.pred_units[l](errors[l], r_states[l], c_states[l])
            else:
                upsamp_error = f.upsample(errors[l+1],  scale_factor = 2)
                r_state, c_state = self.pred_units[l](upsamp_error, r_states[l], c_states[l])
            #get new Ahats's
            ahat = ahat_units[l](r_state)
            #use to calculate errors:
            if l == 0:
                ahat = f.min(ahat, 1) #do satlu
                error = self.error_units[l](prediction=ahat, target=input)
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
                error = self.error_units[l](prediction=ahats[l], target = pool_r)
                errors[l] = error

            if l < self.num_of_layers - 1:
                r_state, c_state = self.pred_units[l](error, r_states[l], c_states[l])
                r_states[l] = r_state
                c_states[l] = c_state

            if training:
                layer_error = f.mean(f.flatten(errors[l]), dim=1, keepdim=True) ## Not sure if dim should be 1 or -1?? pytorch is confusing.
                all_error = layer_error if l == 0 else f.cat((all_error, layer_error), dim=1)
                output = all_error

        states = zip(r_states, c_states, errors)

        return output, states
