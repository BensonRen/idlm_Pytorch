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


# The forward model which gives the score for discriminator
class Forward(nn.Module):
    def __init__(self, flags):
        super(Forward, self).__init__()
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):  # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))

        # Conv Layer definitions here
        self.convs = nn.ModuleList([])
        in_channel = 1  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel,
                                                                     flags.conv_kernel_size,
                                                                     flags.conv_stride)):
            if stride == 2:  # We want to double the number
                pad = int(kernel_size / 2 - 1)
            elif stride == 1:  # We want to keep the number unchanged
                pad = int((kernel_size - 1) / 2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")

            self.convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                                 stride=stride, padding=pad))  # To make sure L_out double each time
            in_channel = out_channel  # Update the out_channel

        self.convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out
        # Monitor the gradient list
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            #print(out.size())
            if ind < len(self.linears) - 1:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
            else:
                out = bn(fc(out))
        out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
        # For the conv part
        for ind, conv in enumerate(self.convs):
            out = conv(out)

        S = out.squeeze()
        return S


class Spectra_encoder(nn.Module):
    def __init__(self, flags):
        super(Spectra_encoder, self).__init__()
        """
        This part is the spectra encoder layer definition
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_se = nn.ModuleList([])
        self.bn_linears_se = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_se[0:-1]):               # Excluding the last one as we need intervals
            self.linears_se.append(nn.Linear(fc_num, flags.linear_se[ind + 1]))
            self.bn_linears_se.append(nn.BatchNorm1d(flags.linear_se[ind + 1]))

        # Conv Layer definitions here
        self.convs_se = nn.ModuleList([])
        in_channel = 1                                                  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel_se,
                                                                     flags.conv_kernel_size_se,
                                                                     flags.conv_stride_se)):
            if stride == 2:     # We want to double the number
                pad = int(kernel_size/2 - 1)
            elif stride == 1:   # We want to keep the number unchanged
                pad = int((kernel_size - 1)/2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")
            self.convs_se.append(nn.Conv1d(in_channel, out_channel, kernel_size,
                                          stride=stride, padding=pad))
            in_channel = out_channel  # Update the out_channel
        # Define forward module for separate training
        self.backward_modules = [self.linears_se, self.bn_linears_se, self.convs_se]

    def forward(self, S):
        out = S.unsqueeze(1)
        # For the Conv Layers
        for ind, conv in enumerate(self.convs_se):
            out = conv(out)

        out = out.squeeze()

        # Encode the spectra first into features using linear
        for ind, (fc, bn) in enumerate(zip(self.linears_se, self.bn_linears_se)):
            out = F.relu(bn(fc(out)))
        spec_encode = out
        return spec_encode


class Discriminator(nn.Module):
    def __init__(self, flags):
        super(Discriminator, self).__init__()
        """
        This part is the discriminator model layers definition:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_d = nn.ModuleList([])
        self.bn_linears_d = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_d[0:-1]):               # Excluding the last one as we need intervals
            self.linears_d.append(nn.Linear(fc_num, flags.linear_d[ind + 1]))
            self.bn_linears_d.append(nn.BatchNorm1d(flags.linear_d[ind + 1]))

    def forward(self, G, S):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = torch.cat((G, S), dim=-1)                                                         # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_d, self.bn_linears_d)):
            # print(out.size())
            out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
        return out


class Generator(nn.Module):
    def __init__(self, flags):
        super(Generator, self).__init__()
        """
        This part if the backward model layers definition:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_g = nn.ModuleList([])
        self.bn_linears_g = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_g[0:-1]):               # Excluding the last one as we need intervals
            self.linears_g.append(nn.Linear(fc_num, flags.linear_g[ind + 1]))
            self.bn_linears_g.append(nn.BatchNorm1d(flags.linear_g[ind + 1]))


    def forward(self, spec_encode, z):
        """
        The backward function defines how the backward network is connected
        :param S: The 300-d input spectrum
        :return: G: The 8-d geometry
        """
        # Concatenate the random noise dimension to together with the convoluted spectrum
        out = torch.cat((spec_encode, z), dim=-1)
        
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_g, self.bn_linears_g)):
            out = F.relu(bn(fc(out)))
        G = out
        return G

