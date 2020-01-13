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
        """
        Defining the decoder structures
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.decoder_linears = nn.ModuleList([])
        self.decoder_bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.decoder_linear[0:-1]):  # Excluding the last one as we need intervals
            self.decoder_linears.append(nn.Linear(fc_num, flags.decoder_linear[ind + 1]))
            self.decoder_bn_linears.append(nn.BatchNorm1d(flags.decoder_linear[ind + 1]))

        # Conv Layer definitions here
        self.decoder_convs = nn.ModuleList([])
        in_channel = 1  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.decoder_conv_out_channel,
                                                                     flags.decoder_conv_kernel_size,
                                                                     flags.decoder_conv_stride)):
            if stride == 2:  # We want to double the number
                pad = int(kernel_size / 2 - 1)
            elif stride == 1:  # We want to keep the number unchanged
                pad = int((kernel_size - 1) / 2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")

            self.decoder_convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                                 stride=stride, padding=pad))  # To make sure L_out double each time
            in_channel = out_channel  # Update the out_channel

        self.decoder_convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))
        """
        Defining the encoder structure
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.encoder_linears = nn.ModuleList([])
        self.encoder_bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.encoder_linear[0:-1]):  # Excluding the last one as we need intervals
            self.encoder_linears.append(nn.Linear(fc_num, flags.encoder_linear[ind + 1]))
            self.encoder_bn_linears.append(nn.BatchNorm1d(flags.encoder_linear[ind + 1]))

        # Conv Layer definitions here
        self.encoder_convs = nn.ModuleList([])
        in_channel = 1  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.encoder_conv_out_channel,
                                                                     flags.encoder_conv_kernel_size,
                                                                     flags.encoder_conv_stride)):
            if stride == 2:  # We want to double the number
                pad = int(kernel_size / 2 - 1)
            elif stride == 1:  # We want to keep the number unchanged
                pad = int((kernel_size - 1) / 2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")
            self.encoder_convs.append(nn.Conv1d(in_channel, out_channel, kernel_size,
                                          stride=stride, padding=pad))
            in_channel = out_channel  # Update the out_channel
        self.encoder_convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

    def encoder(self, y):
        """
        The decoder part of the auto encoder
        :param y: The spectra to be encoded
        :return: The encoded spectra (dimension reduced)
        """
        out = y.unsqueeze(1)
        # For the Conv Layers
        for ind, conv in enumerate(self.encoder_convs):
            out = conv(out)
        out = out.squeeze()
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.encoder_linears, self.encoder_bn_linears)):
            out = F.relu(bn(fc(out)))
        return out

    def decoder(self, y_code):
        """
        The encoder part of the auto encoder
        :param y_code: The coded spectra (dimension reduced)
        :return: The decoded original spectra
        """
        out = y_code
        for ind, (fc, bn) in enumerate(zip(self.decoder_linears, self.decoder_bn_linears)):
            out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear

        # The normal mode to train without Lorentz
        out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
        # For the conv part
        for ind, conv in enumerate(self.decoder_convs):
            out = conv(out)
        return out.squeeze()

    def forward(self, y):
        """
        The Forward function of an auto encoder
        :param y: The raw spectrum input
        :return: The encoded spectrum output
        """
        return self.decoder(self.encoder(y))