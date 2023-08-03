# London Bielicke

import torch
import numpy as np

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ torch.nn.Tanh() for l in  self.layers ]
        # self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

        self.feauture_layer = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh()
        )

        self.readout = torch.nn.Linear(128, self.output_size)

        self.adv_output_size = 1
        self.adv_readout = torch.nn.Linear(self.hidden_sizes[-1], self.adv_output_size)

    def forward(self, x):
        # for layer,activation_fun in zip(self.layers, self.activations):
        #     x = activation_fun(layer(x))

        x = self.feauture_layer(x)

        return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()

class DuelingDQN(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_size
        self.output_dim = output_size

        self.feauture_layer = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh()
        )

        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
