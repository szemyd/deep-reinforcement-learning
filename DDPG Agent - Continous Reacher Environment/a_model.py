import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_space, action_space, random_seed = 32, hidden_layer_param = [400, 300] ):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(random_seed)

        new_hidden_layer_param = hidden_layer_param.copy()
        self.fc_in = nn.Linear(state_space, new_hidden_layer_param[0])

        if len(new_hidden_layer_param) < 2: self.hidden_layers = []
        else: self.hidden_layers = nn.ModuleList([nn.Linear(new_hidden_layer_param[i], new_hidden_layer_param[i+1]) for i in range(len(new_hidden_layer_param)-1)])
        
        self.fc_out = nn.Linear(new_hidden_layer_param[-1], action_space)

        self.activation = F.relu

        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_in.weight.data.uniform_(*hidden_init(self.fc_in))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc_in(state))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        x = self.fc_out(x)

        return torch.tanh(x)

class Critic(nn.Module):
    def __init__(self, state_space, action_space, random_seed = 32, hidden_layer_param = [400, 300] ):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(random_seed)

        new_hidden_layer_param = hidden_layer_param.copy()
        self.fc_in = nn.Linear(state_space, new_hidden_layer_param[0])

        new_hidden_layer_param[0]+=action_space # this has to be under fc_in
        
        if len(new_hidden_layer_param) < 2: self.hidden_layers = []
        else: self.hidden_layers = nn.ModuleList([nn.Linear(new_hidden_layer_param[i], new_hidden_layer_param[i+1]) for i in range(len(new_hidden_layer_param)-1)])
        # Critic throws back a single value (output = 1), which is the estimated value of the given state # 
        self.fc_out = nn.Linear(new_hidden_layer_param[-1], 1)

        self.activation = F.relu

        self.reset_parameters()

    def reset_parameters(self):
        self.fc_in.weight.data.uniform_(*hidden_init(self.fc_in))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fc_in(state))
        x = torch.cat((xs, action), dim=1)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        return self.fc_out(x)