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
from torch import pow, add, mul, div, sqrt


class Forward(nn.Module):
    def __init__(self, flags):
        super(Forward, self).__init__()
        # Set up whether this uses a Lorentzian oscillator, this is a boolean value
        self.use_lorentz = flags.use_lorentz

        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))

        # Assert the last entry of the fc_num is a multiple of 3 (This is for Lorentzian part)
        if flags.use_lorentz:
            self.num_spec_point = 300
            assert flags.linear[-1] % 3 == 0, "Please make sure your last layer in linear is\
                                                multiple of 3 since you are using lorentzian"
            # Set the number of lorentz oscillator
            self.num_lorentz = int(flags.linear[-1]/3)

            # Create the constant for mapping the frequency w
            w_numpy = np.arange(0.8, 1.5, self.num_spec_point)

            # Create the tensor from numpy array
            cuda = True if torch.cuda.is_available() else False
            if cuda:
                self.w = torch.tensor(w_numpy).cuda()
            else:
                self.w = torch.tensor(w_numpy)

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

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            #print(out.size())
            out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear

        # If use lorentzian layer, pass this output to the lorentzian layer
        if self.use_lorentz:
            # Get the out into (batch_size, num_lorentz, 3)
            out = out.view([-1, int(out.size(1)/3), 3])

            # Get the list of params for lorentz, also add one extra dimension at 3rd one to
            w0 = out[:, :, 0].unsqueeze(2)
            wp = out[:, :, 1].unsqueeze(2)
            g  = out[:, :, 2].unsqueeze(2)
            #print("W0 size:", w0.size())

            # Expand them to the make the parallelism, (batch_size, #Lor, #spec_point)
            w0 = w0.expand(out.size(0), self.num_lorentz, self.num_spec_point)
            wp = wp.expand(out.size(0), self.num_lorentz, self.num_spec_point)
            g  = g.expand(out.size(0), self.num_lorentz, self.num_spec_point)
            w_expand = self.w.expand_as(g)

            # Get the powers first
            w02 = pow(w0, 2)
            wp2 = pow(wp, 2)
            w2 = pow(w_expand, 2)
            g2 = pow(g, 2)

            # Start calculating
            s1 = add(w02, -w2)
            s12= pow(s1, 2)
            n1 = mul(wp2, s1)
            n2 = mul(wp2, mul(w_expand, g))
            denom = add(s12, mul(w2, g2))
            e1 = div(n1, denom)
            e2 = div(n2, denom)
            e12 = pow(e1, 2)
            e22 = pow(e2, 2)
            
            n = sqrt(0.5* add(sqrt(add(e12, e22)), e1))
            k = sqrt(0.5* add(sqrt(add(e12, e22)), -e1))
            n_12 = pow(n+1, 2)
            k2 = pow(k, 2)

            T = div(4*n, add(n_12, k2))
            # Last step, sum up except for the 0th dimension of batch_size
            T = torch.sum(T, 1).float()
            #print("Type of T is:", T.dtype)
            return T

        # The normal mode to train without Lorentz
        out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
        # For the conv part
        for ind, conv in enumerate(self.convs):
            #print(out.size())
            out = conv(out)

        # Final touch, because the input is normalized to [-1,1]
        # S = tanh(out.squeeze())
        # print(S.size())
        S = out.squeeze()
        print("Type of R is:", type(S))
        return S

    def lorentz_layer(self, S):
        """
        Lorentzian oscillator function takes the input tensor S, and uses the elements in S
        as parameters for a series of Lorentz oscillators, in groups of 3. Currently uses LO
        model to calculate and return transmission tensor T to compare to simulations.
        :param S: The previous Spectra output of the model if not using Lorentzian modeule
        :return:
        """

        T = torch.zeros([S.size(0), S.size(1)], dtype=torch.float32, requires_grad=True)

        # For each examples in the batch
        for i in range(S.size(0)):
            print("i = ", i)

            # For the 300 dimension output
            for j in range(S.size(1)):
                # print("j = ", j)

                # For each of the 100 Lorentzian oscillators
                for k in range(0, S.size(1), 3):
                    w = 0.8 + j / S.size(1)
                    wp = S[i, k]
                    w0 = S[i, k + 1]
                    g = S[i, k + 2]

                    # e1 = (wp ** 2) * (w0 ** 2 - w ** 2) / ((w0 ** 2 - w ** 2) ** 2 + (w ** 2) * (g ** 2))
                    # e2 = (wp ** 2) * (w * g) / ((w0 ** 2 - w ** 2) ** 2 + (w ** 2) * (g ** 2))
                    #
                    # n = (0.5 * (e1 ** 2 + e2 ** 2) ** 0.5 + 0.5 * e1) ** 0.5
                    # k = (0.5 * (e1 ** 2 + e2 ** 2) ** 0.5 - 0.5 * e1) ** 0.5
                    #
                    # dT = (4 * n) / ((n + 1) ** 2 + k ** 2)

                    """
                    Ben's debugging code, keep it here before this function runs smoothly
                    print("w0 type", type(w0))
                    print("wp type", type(wp))
                    print("w type", type(w))
                    print("g type", type(g))
                    """

                    sub1 = add(pow(w0, 2), - math.pow(w, 2))
                    num1 = mul(pow(wp, 2), sub1)
                    num2 = mul(pow(wp, 2), mul(w, g))
                    denom = add(pow(sub1, 2),
                                      mul(math.pow(w, 2), pow(g, 2)))
                    e1 = div(num1, denom)
                    e2 = div(num2, denom)

                    n = sqrt(add(0.5 * add(pow(e1, 2), pow(e2, 2)), 0.5 * e1))
                    sub2 = add(0.5 * add(pow(e1, 2), pow(e2, 2)), -0.5 * e1)
                    k = sqrt(sub2)

                    dT = div(4 * n, add(pow(add(n, 1), 2), math.pow(k, 2)))
                    T.data[i, j] += dT.data

        return T

