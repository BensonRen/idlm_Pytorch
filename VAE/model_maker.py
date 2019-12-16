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


class Decoder(nn.Module):
    def __init__(self, flags):
        super(Decoder, self).__init__()
        """
        This part is the Decoder model layers definition:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_D = nn.ModuleList([])
        self.bn_linears_D = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_D[0:-1]):               # Excluding the last one as we need intervals
            self.linears_D.append(nn.Linear(fc_num, flags.linear_D[ind + 1]))
            self.bn_linears_D.append(nn.BatchNorm1d(flags.linear_D[ind + 1]))

    def forward(self, z, S_enc):
        """
        The forward function which defines how the network is connected
        :param S_enc:  The encoded spectra input
        :return: G: Geometry output
        """
        out = torch.concatenate(z, S_enc)                                                         # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_D, self.bn_linears_D)):
            # print(out.size())
            out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
        return out


class Encoder(nn.Module):
    def __init__(self, flags):
        super(Encoder, self).__init__()
        """
        This part is the Decoder model layers definition:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_E = nn.ModuleList([])
        self.bn_linears_E = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_E[0:-1]):  # Excluding the last one as we need intervals
            self.linears_E.append(nn.Linear(fc_num, flags.linear_E[ind + 1]))
            self.bn_linears_E.append(nn.BatchNorm1d(flags.linear_E[ind + 1]))

        # Re-parameterization
        self.zmean_layer = nn.Linear(flags.linear_E[-1], flags.dim_latent_z)
        self.z_log_var_layer = nn.Linear(flags.linear_E[-1], flags.dim_latent_z)

    def forward(self, z, S_enc):
        """
        The forward function which defines how the network is connected
        :param S_enc:  The encoded spectra input
        :return: G: Geometry output
        """
        out = torch.concatenate(z, S_enc)  # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_E, self.bn_linears_E)):
            # print(out.size())
            out = F.relu(bn(fc(out)))  # ReLU + BN + Linear
        z_mean = self.zmean_layer(out)
        z_log_var = self.z_log_var_layer(out)
        return z_mean, z_log_var

class SpectraEncoder(nn.Module):
    def __init__(self, flags):
        super(SpectraEncoder, self).__init__()
        """
        This part if the backward model layers definition:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears_SE = nn.ModuleList([])
        self.bn_linears_SE = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_SE[0:-1]):               # Excluding the last one as we need intervals
            self.linears_SE.append(nn.Linear(fc_num, flags.linear_SE[ind + 1]))
            self.bn_linears_SE.append(nn.BatchNorm1d(flags.linear_SE[ind + 1]))

        # Conv Layer definitions here
        self.convs_SE = nn.ModuleList([])
        in_channel = 1                                                  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel_SE,
                                                                     flags.conv_kernel_size_SE,
                                                                     flags.conv_stride_SE)):
            if stride == 2:     # We want to double the number
                pad = int(kernel_size/2 - 1)
            elif stride == 1:   # We want to keep the number unchanged
                pad = int((kernel_size - 1)/2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")
            self.convs_SE.append(nn.Conv1d(in_channel, out_channel, kernel_size,
                                          stride=stride, padding=pad))
            in_channel = out_channel  # Update the out_channel

    def forward(self, S):
        """
        The backward function defines how the backward network is connected
        :param S: The 300-d input spectrum
        :return: S_enc: The n-d output encoded spectrum
        """
        out = S.unsqueeze(1)
        # For the Conv Layers
        for ind, conv in enumerate(self.convs_SE):
            out = conv(out)

        out = out.squeeze()
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears_SE, self.bn_linears_SE)):
            out = F.relu(bn(fc(out)))
        S_enc = out
        return S_enc
