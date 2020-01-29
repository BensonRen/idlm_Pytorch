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
    def __init__(self, flags, fre_low=0.8, fre_high=1.5):
        super(Forward, self).__init__()

        # Set up whether this uses a Lorentzian oscillator, this is a boolean value
        self.use_lorentz = flags.use_lorentz
        self.use_conv = flags.use_conv

        # Assert the last entry of the fc_num is a multiple of 3 (This is for Lorentzian part)
        if flags.use_lorentz:
            # there is 1 extra parameter for lorentzian setting for epsilon_inf
            flags.linear[-1] += 1

            self.num_spec_point = 300
            assert (flags.linear[-1] - 1) % 3 == 0, "Please make sure your last layer in linear is\
                                                        multiple of 3 (+1) since you are using lorentzian"
            # Set the number of lorentz oscillator
            self.num_lorentz = int(flags.linear[-1] / 3)

            # Create the constant for mapping the frequency w
            w_numpy = np.arange(fre_low, fre_high, (fre_high - fre_low) / self.num_spec_point)

            self.fix_w0 = flags.fix_w0
            self.w0 = torch.tensor(np.arange(0, 5, 5 / self.num_lorentz))

            # Create the tensor from numpy array
            cuda = True if torch.cuda.is_available() else False
            if cuda:
                self.w = torch.tensor(w_numpy).cuda()
                self.w0 = self.w0.cuda()
            else:
                self.w = torch.tensor(w_numpy)

        """
        General layer definitions:
        """
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
        # Monitor the gradient list
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            #print(out.size())
            if ind < len(self.linears) - 1:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
            else:
                out = bn(fc(out))

        # If use lorentzian layer, pass this output to the lorentzian layer
        if self.use_lorentz:
            out = torch.sigmoid(out)            # Lets say w0, wp is in range (0,5) for now
            #out = F.relu(out) + 0.00001

            # Get the out into (batch_size, num_lorentz, 3) and the last epsilon_inf baseline
            epsilon_inf = out[:, -1]
            out = out[:, 0:-1].view([-1, int(out.size(1)/3), 3])

            # Get the list of params for lorentz, also add one extra dimension at 3rd one to
            if self.fix_w0:
                w0 = self.w0.unsqueeze(0).unsqueeze(2)
            else:
                w0 = out[:, :, 0].unsqueeze(2) * 1.5
            wp = out[:, :, 1].unsqueeze(2) * 5
            g  = out[:, :, 2].unsqueeze(2) * 0.05

            # This is for debugging purpose (Very slow), recording the output tensors
            self.w0s = w0.data.cpu().numpy()
            self.wps = wp.data.cpu().numpy()
            self.gs = g.data.cpu().numpy()
            self.eps_inf = epsilon_inf.data.cpu().numpy()

            # Expand them to the make the parallelism, (batch_size, #Lor, #spec_point)
            w0 = w0.expand(out.size(0), self.num_lorentz, self.num_spec_point)
            wp = wp.expand_as(w0)
            g = g.expand_as(w0)
            w_expand = self.w.expand_as(g)
            """
            Testing code
            #print("c1 size", self.c1.size())
            #print("w0 size", w0.size())
            End of testing module
            """

            """
            # This is version that easier to understand by human, but memory intensive
            # Therefore we implement the less memory aggressive one below, make sure you use only one
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
            """
            # This is the version of more "machine" code that hard to understand but much more memory efficient
            e1 = div(mul(pow(wp, 2), add(pow(w0, 2), -pow(w_expand, 2))), add(pow(add(pow(w0, 2), -pow(w_expand, 2)), 2), mul(pow(w_expand, 2), pow(g, 2))))
            e2 = div(mul(pow(wp, 2), mul(w_expand, pow(g, 2))), add(pow(add(pow(w0, 2), -pow(w_expand, 2)), 2), mul(pow(w_expand, 2), pow(g, 2))))
            # End line for the 2 versions of code that do the same thing, 1 for memory efficient but ugly

            self.e2 = e2.data.cpu().numpy()                 # This is for plotting the imaginary part
            self.e1 = e1.data.cpu().numpy()                 # This is for plotting the imaginary part
            """
            debugging purposes: 2019.12.10 Bens code for debugging the addition of epsilon_inf
            print("size of e1", e1.size())
            print("size pf epsilon_inf", epsilon_inf.size())
            """

            """
            # This is version that easier to understand by human, but memory intensive
            # Therefore we implement the less memory aggressive one below, make sure you use only one
            # the correct calculation should be adding up the es
            e1 = torch.sum(e1, 1)
            e2 = torch.sum(e2, 1)

            epsilon_inf = epsilon_inf.unsqueeze(1).expand_as(e1)        #Change the shape of the epsilon_inf

            e1 += epsilon_inf

            # print("e1 size", e1.size())
            # print("e2 size", e2.size())
            e12 = pow(e1, 2)
            e22 = pow(e2, 2)

            n = sqrt(0.5 * add(sqrt(add(e12, e22)), e1))
            k = sqrt(0.5 * add(sqrt(add(e12, e22)), -e1))
            n_12 = pow(n+1, 2)
            k2 = pow(k, 2)
            T = div(4*n, add(n_12, k2)).float()
            """
            # This is the memory efficient version of code
            n = sqrt(0.5 * add(sqrt(add(pow(torch.sum(e1, 1), 2), pow(torch.sum(e2, 1), 2))), torch.sum(e1, 1)))
            k = sqrt(0.5 * add(sqrt(add(pow(torch.sum(e1, 1), 2), pow(torch.sum(e2, 1), 2))), -torch.sum(e1, 1)))
            T = div(4*n, add(pow(n+1, 2), pow(k, 2))).float()
            # End line for 2 versions of code that do the same thing, 1 for memory efficient but ugly

            """
            Debugging and plotting (This is very slow, comment to boost)
            """
            self.T_each_lor = T.data.cpu().numpy()          # This is for plotting the transmittion
            self.N = n.data.cpu().numpy()                 # This is for plotting the imaginary part
            self.K = k.data.cpu().numpy()                 # This is for plotting the imaginary part

            # print("T size",T.size())
            # Last step, sum up except for the 0th dimension of batch_size (deprecated since we sum at e above)
            # T = torch.sum(T, 1).float()
            return T

        # The normal mode to train without Lorentz
        if self.use_conv:
            out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
            # For the conv part
            for ind, conv in enumerate(self.convs):
                out = conv(out)

            # Final touch, because the input is normalized to [-1,1]
            # S = tanh(out.squeeze())
            out = out.squeeze()
        return out

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

