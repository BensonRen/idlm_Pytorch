"""
This is the auto-encoder module where the model is defined. It uses the nn.Module as backbone to create the network structure
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

class AutoEncoder(nn.Module):
    """
    The Auto Encoder that shrinks the dimension of the spectra into lower than 8d
    Once it learns the encoder-decoder relationship using fc and conv layers, it is fixed.
    Therefore it is of interest to pre-train the encoder and fix that
    """
    def __init__(self, flags):
        super(AutoEncoder, self).__init__()

        # Linear Layer and Batch_norm Layer definitions here
        self.decoder_linears = nn.ModuleList([])
        self.decoder_bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):  # Excluding the last one as we need intervals
            self.decoder_linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.decoder_bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))

        # Conv Layer definitions here
        self.decoder_convs = nn.ModuleList([])
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

            self.encoder_convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                                 stride=stride, padding=pad))  # To make sure L_out double each time
            in_channel = out_channel  # Update the out_channel

        self.decoder_convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

        # Linear Layer and Batch_norm Layer definitions here
        self.linears_b = nn.ModuleList([])
        self.bn_linears_b = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear_b[0:-1]):  # Excluding the last one as we need intervals
            self.linears_b.append(nn.Linear(fc_num, flags.linear_b[ind + 1]))
            self.bn_linears_b.append(nn.BatchNorm1d(flags.linear_b[ind + 1]))

        # Conv Layer definitions here
        self.convs_b = nn.ModuleList([])
        in_channel = 1  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel_b,
                                                                     flags.conv_kernel_size_b,
                                                                     flags.conv_stride_b)):
            if stride == 2:  # We want to double the number
                pad = int(kernel_size / 2 - 1)
            elif stride == 1:  # We want to keep the number unchanged
                pad = int((kernel_size - 1) / 2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")
            self.convs_b.append(nn.Conv1d(in_channel, out_channel, kernel_size,
                                          stride=stride, padding=pad))
            in_channel = out_channel  # Update the out_channel


    def encoder(self, y):
        """
        The decoder part of the auto encoder
        :param y: The spectra to be encoded
        :return: The encoded spectra (dimension reduced)
        """

    def decoder(self, y_code):
        """
        The encoder part of the auto encoder
        :param y_code: The coded spectra (dimension reduced)
        :return: The decoded original spectra
        """
