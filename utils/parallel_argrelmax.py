"""
This is a small function for implementing the torch version (parallel) of np.argrelmax function, requested by Omar
Ben Ren, 2019.04.16
It simply uses 2 convolution and relu with one multiply in the end to get the index of local minima
Note that this local minima has to be strictly larger than both sides (no flat area is allowed)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
def torch_argrelmax(input_tensor):
    up_kernel = torch.tensor([-1, 1, 0]).view(1,1,-1)
    down_kernel = torch.tensor([0, 1, -1]).view(1,1,-1)
    up_branch = F.conv1d(input=input_tensor, weight=up_kernel, stride=1, bias=None, padding=1)
    down_branch = F.conv1d(input=input_tensor, weight=down_kernel, stride=1, bias=None, padding=1)
    return 1 * (F.relu(up_branch) * F.relu(down_branch) != 0)


if __name__ == '__main__':
    a = torch.tensor([0,10,20,30,20,10,20,10,20,30,40,50,1]).view(1,1,-1)
    print(a)
    print(torch_argrelmax(a))