# This is the model maker for the Invertible Neural Network

# From Built-in
from time import time

# From libs
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# From FrEIA
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom


def cINN(flags):
    """
    The constructor for INN network
    :param flags: input flags from configuration
    :return: The INN network
    """
    # Set up the conditional node (y)
    cond_node = ConditionNode(flags.dim_y)
    # Start from input layer
    nodes = [InputNode(flags.dim_x, name='input')]
    # Recursively add the coupling layers and random permutation layer
    for i in range(flags.couple_layer_num):
        nodes.append(Node(nodes[-1], GLOWCouplingBlock,
                          {'subnet_constructor': subnet_fc,
                           'clamp': 2.0},
                          conditions=cond_node,
                          name='coupling_{}'.format(i)))
        nodes.append(Node(nodes[-1], PermuteRandom, {'seed': i}, name='permute_{}'.format(i)))
    # Attach the output Node
    nodes.append(OutputNode(nodes[-1], name='output'))
    nodes.append(cond_node)
    print("The nodes are:", nodes)
    # Return the
    return ReversibleGraphNet(nodes, verbose=True)

##########
# Subnet #
##########

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(), 
                                  nn.Linear(512, 512), nn.ReLU(),
                                  nn.Linear(512,  c_out))

# Can only take 2 input.... Due to the restriction of the framework used

"""
class subnet_fc(nn.Module):
    ""
    Construct the subnet module
    :param flags: input flags from configuration
    :return: The sub net used in affine layers
    ""
    def __init__(self, flags):
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.subnet_linear[0: -1]):
            self.linears.append(nn.Linear(fc_num, flags.subnet_linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.subnet_linear[ind + 1]))


    def forward(self, out):
        ""
        The forward function of the subnet structure
        :param out: The input of the subnet structure
        :return: Output of subnet structure
        ""
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if ind != len(self.linears) - 1:
                out = F.relu(bn(fc(out)))
            else:
                out = fc(out)
        return out
"""
