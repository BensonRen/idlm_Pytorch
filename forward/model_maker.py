"""
This is the module where the model is defined. It uses the nn.Module as backbone to create the network structure
"""
# Own modules

# Built in
import math
# Libs

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch

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

        out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
        # For the conv part
        for ind, conv in enumerate(self.convs):
            #print(out.size())
            out = conv(out)

        # Final touch, because the input is normalized to [-1,1]
        # S = tanh(out.squeeze())
        # print(S.size())
        S = out.squeeze()

        # If use lorentzian layer, pass this output to the lorentzian layer
        if self.use_lorentz:
            S = self.lorentz_layer(S)

        return S

    def lorentz_layer(self, S):
        """
        Lorentzian oscillator function takes the input tensor S, and uses the elements in S
        as parameters for a series of Lorentz oscillators, in groups of 3. Currently uses LO
        model to calculate and return transmission tensor T to compare to simulations.
        :param S: The previous Spectra output of the model if not using Lorentzian modeule
        :return:
        """

        T = torch.zeros([S.size(0), S.size(1)], dtype=torch.float32)

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

                    sub1 = torch.add(torch.pow(w0, 2), - math.pow(w, 2))
                    num1 = torch.mul(torch.pow(wp, 2), sub1)
                    num2 = torch.mul(torch.pow(wp, 2), torch.mul(w, g))
                    denom = torch.add(torch.pow(sub1, 2),
                                      torch.mul(math.pow(w, 2), torch.pow(g, 2)))
                    e1 = torch.div(num1, denom)
                    e2 = torch.div(num2, denom)

                    n = torch.sqrt(torch.add(0.5 * torch.add(torch.pow(e1, 2), torch.pow(e2, 2)), 0.5 * e1))
                    sub2 = torch.add(0.5 * torch.add(torch.pow(e1, 2), torch.pow(e2, 2)), -0.5 * e1)
                    k = torch.sqrt(sub2)

                    dT = torch.div(4 * n, torch.add(torch.pow(torch.add(n, 1), 2), math.pow(k, 2)))
                    T.data[i, j] += dT.data

        return T
