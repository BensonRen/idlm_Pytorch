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

class Forward(nn.Module):
    def __init__(self, flags):
        super(Forward, self).__init__()
        """
        This part is the forward model layers definition:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_f = nn.ModuleList([])
        self.bn_linears_f = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_f[0:-1]):               # Excluding the last one as we need intervals
            self.linears_f.append(nn.Linear(fc_num, flags.linear_f[ind + 1]))
            self.bn_linears_f.append(nn.BatchNorm1d(flags.linear_f[ind + 1]))

        # Conv Layer definitions here
        self.convs_f = nn.ModuleList([])
        in_channel = 1                                                  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel_f,
                                                                     flags.conv_kernel_size_f,
                                                                     flags.conv_stride_f)):
            if stride == 2:     # We want to double the number
                pad = int(kernel_size/2 - 1)
            elif stride == 1:   # We want to keep the number unchanged
                pad = int((kernel_size - 1)/2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")

            self.convs_f.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                stride=stride, padding=pad)) # To make sure L_out double each time
            in_channel = out_channel # Update the out_channel
        # Get the channel number down
        self.convs_f.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_f, self.bn_linears_f)):
            # print(out.size())
            out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
        if self.convs_f:
            # The normal mode to train without Lorentz
            out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
            # For the conv part
            for ind, conv in enumerate(self.convs_f):
                out = conv(out)
            out = out.squeeze()
        return out


class Backward(nn.Module):
    def __init__(self, flags):
        super(Backward, self).__init__()
        """
        This part if the backward model layers definition:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_b = nn.ModuleList([])
        self.bn_linears_b = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_b[0:-1]):               # Excluding the last one as we need intervals
            self.linears_b.append(nn.Linear(fc_num, flags.linear_b[ind + 1]))
            self.bn_linears_b.append(nn.BatchNorm1d(flags.linear_b[ind + 1]))

        # Conv Layer definitions here
        self.convs_b = nn.ModuleList([])
        in_channel = 1                                                  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel_b,
                                                                     flags.conv_kernel_size_b,
                                                                     flags.conv_stride_b)):
            if stride == 2:     # We want to double the number
                pad = int(kernel_size/2 - 1)
            elif stride == 1:   # We want to keep the number unchanged
                pad = int((kernel_size - 1)/2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")
            self.convs_b.append(nn.Conv1d(in_channel, out_channel, kernel_size,
                                          stride=stride, padding=pad))
            in_channel = out_channel  # Update the out_channel
        if len(self.convs_b):  # Make sure there is not en empty one
            self.convs_b.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

    def forward(self, S):
        """
        The backward function defines how the backward network is connected
        :param S: The 300-d input spectrum
        :return: G: The 8-d geometry
        """
        out = S
        if self.convs_b:
            out = out.unsqueeze(1)
            # For the Conv Layers
            for ind, conv in enumerate(self.convs_b):
                out = conv(out)

            out = out.squeeze(1)
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_b, self.bn_linears_b)):
            if ind != len(self.linears) - 1:
                out = F.relu(bn(fc(out)))
            else:
                out = fc(out)
        G = out
        return G

"""
    def forward(self, G_in=None, S_in=None, forward_model=False):
        ""
        The forward function of the whole model. 2 modes supported for forward module training and full model training
        :param G_in: The input Geometery for forward training
        :param S_in: The input Spectra for full model training
        :param forward_model: Boolean flag for whether only use forward model training or not
        :return:
        ""

        ""
        # Checking some invariant at testing
        if forward_model:
            assert (G_in is not None and S_in is None), \
                "Forward_model training, G_in can not be None and S_in should be None"
        else:
            assert (G_in is None and S_in is not None), \
                "Full model training, S_in can not be None and G_in should be None"
        ""

        if forward_model:
            G_out = G_in
            S_out = forward_model(G_in)
        else:
            G_out = backward_model(S_in)
            S_out = forward_model(G_out)
        return G_out, S_out
"""


