# London Bielicke

import torch
import numpy as np

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fun=torch.nn.Tanh(), output_activation=None ):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ activation_fun for l in  self.layers ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

        # self.feauture_layer = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_size, 256),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(256, 128),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(128, 128),
        #     torch.nn.Tanh()
        # )

        # self.readout = torch.nn.Linear(128, self.output_size)

        # self.adv_output_size = 1
        # self.adv_readout = torch.nn.Linear(self.hidden_sizes[-1], self.adv_output_size)

    def forward(self, x):
        # for layer,activation_fun in zip(self.layers, self.activations):
        #     x = activation_fun(layer(x))

        # x = self.feauture_layer(x)

        # return self.readout(x)
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

class DuelingDQN(torch.nn.Module):

    def __init__(self, input_size, hidden_sizes, hidden_sizes_values, hidden_sizes_advantages, output_size, activation_fun=torch.nn.Tanh(),activation_fun_values=torch.nn.ReLU(),activation_fun_advantages=torch.nn.ReLU(), output_activation=None ):
        super(DuelingDQN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        layer_sizes_values = [layer_sizes[-1]] + hidden_sizes_values
        layer_sizes_advantages = [layer_sizes[-1]] + hidden_sizes_advantages
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.layers_values = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes_values[:-1], layer_sizes_values[1:])])
        self.layers_advantages = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes_advantages[:-1], layer_sizes_advantages[1:])])
        self.activations = [ activation_fun for l in  self.layers ]
        self.activation_fun_values = [ activation_fun_values for l in  self.layers_values ]
        self.activation_fun_advantages = [ activation_fun_advantages for l in  self.layers_advantages ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)


        # self.feauture_layer = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_dim, 256),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(256, 128),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(128, 128),
        #     torch.nn.Tanh()
        # )

        # self.value_stream = torch.nn.Sequential(
        #     torch.nn.Linear(128, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 1)
        # )

        # self.advantage_stream = torch.nn.Sequential(
        #     torch.nn.Linear(128, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, self.output_dim)
        # )

    def forward(self, state):
        # features = self.feauture_layer(state)
        # values = self.value_stream(features)
        # advantages = self.advantage_stream(features)
        # return values + (advantages - advantages.mean())
        
    
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        features = x.copy()
        
        for layer,activation_fun in zip(self.layers_values, self.activation_fun_value):
            x = activation_fun(layer(x))
        values = x 

        for layer,activation_fun in zip(self.layers_advantages, self.activation_fun_advantage):
            features = activation_fun(layer(features))
        advantages = features 
        
        return values + (advantages - advantages.mean())

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
