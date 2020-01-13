"""
This is the module where the model is defined. It uses the nn.Module as backbone to create the network structure
"""
# Own modules

# Built in
import math
# Libs
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import add, mul, exp


class CouplingLayer(nn.Module):
    """
    The Coupling Layer of the INN, additive
    """
    def __init__(self, flags, orientation=True):
        super(CouplingLayer, self).__init__()
        self.orientation = orientation                  # The orientation of this, True means positive and false means inversed
        self.leakyrelu = nn.LeakyReLU(flags.leakyrelu_slope)
        self.halflen = flags.linear[0]//2                      # where we split
        self.linear_layers_s12t12 = []                # Define the list for s12t12
        """
        There would be 4 lists inside this linear_layers_s12t12, indicating 
        s1,s2,t1,t2 4 complicated functions which can be modelled using fc nn with leakyrelu
        each list has 2 components: linears, bn_linears
        """
        for i in range(4):
            # Linear layer and batch norm layer for s1,s2,t1,t2 function
            linears = nn.ModuleList([])
            bn_linears = nn.ModuleList([])

            for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
                linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
                bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))
            self.linear_layers_s12t12.append([linears, bn_linears])

    def f(self, x, fname):
        out = x
        lb_pair = None                              # Initialize linear batch norm pair
        if fname == 's1':
            lb_pair = self.linear_layers_s12t12[0]
        elif fname == 's2':
            lb_pair = self.linear_layers_s12t12[1]
        elif fname == 't1':
            lb_pair = self.linear_layers_s12t12[2]
        elif fname == 't2':
            lb_pair = self.linear_layers_s12t12[3]
        else:
            raise ValueError('Please specify your f function to be either one of the '
                             'followings: s1, s2, t1, t2 as illustrated in the original paper')
        for ind, (fc, bn) in enumerate(zip(*lb_pair)):
            out = self.leakyrelu(bn(fc(out)))  # ReLU + BN + Linear

    def forward(self, x, logdet, invert=False):
        if invert:
            if self.orientation:
                v1, v2 = x[:self.halflen], x[self.halflen:]
            else:
                v1, v2 = x[self.halflen:], x[:self.halflen]
            u2 = mul(add(v2, -self.f(v1, 't1')), exp(-self.f(v1, 's1')))
            u1 = mul(add(v1, -self.f(v2, 't2')), exp(-self.f(u2, 's2')))
            return torch.cat((u1, u2), axis=1), logdet              # return the concated one
        else:
            if self.orientation:
                u1, u2 = x[:self.halflen], x[self.halflen:]
            else:
                u1, u2 = x[self.halflen:], x[:self.halflen]
            v1 = add(mul(u1, exp(self.f(u2, 's2'))), self.f(u2, 't2'))
            v2 = add(mul(u2, exp(self.f(v1, 's1'))), self.f(v1, 't1'))
            return torch.cat((v1, v2), axis=1), logdet              # return the concatenated one


class INN(nn.Module):
    def __init__(self, flags):
        super(INN, self).__init__()
    """ #For this stage we are not construting the INN model itself
    
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))

        # Conv Layer definitions here
        self.convs = nn.ModuleList([])
        in_channel = 1                                                  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel,
                                                                     flags.conv_kernel_size,
                                                                     flags.conv_stride)):
            if stride == 2:     # We want to double the number
                pad = int(kernel_size/2 - 1)
            elif stride == 1:   # We want to keep the number unchanged
                pad = int((kernel_size - 1)/2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")

            self.convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                stride=stride, padding=pad)) # To make sure L_out double each time
            in_channel = out_channel # Update the out_channel

        self.convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))
    """
    def randomize_geometry_eval(self):
        self.geometry_eval = torch.randn_like(self.geometry_eval, requires_grad=True)       # Randomize

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out
        if self.bp:                                               # If the evaluation mode
            out = self.geometry_eval
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            # print(out.size())
            out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear

        # The normal mode to train without Lorentz
        out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
        # For the conv part
        for ind, conv in enumerate(self.convs):
            #print(out.size())
            out = conv(out)
        S = out.squeeze()
        return S

