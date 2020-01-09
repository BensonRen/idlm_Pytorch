## This file is snatched https://github.com/DakshIdnani/pytorch-nice
import numpy as np
import torch
import torch.nn as nn

from modules import LogisticDistribution, CouplingLayer, ScalingLayer


class NICE(nn.Module):
  def __init__(self, flags, num_coupling_layers=3):
    super().__init__()

    self.data_dim = flags.data_dim

    # alternating mask orientations for consecutive coupling layers
    masks = [self._get_mask(self.data_dim, orientation=(i % 2 == 0))
                                            for i in range(num_coupling_layers)]

    self.coupling_layers = nn.ModuleList([CouplingLayer(data_dim=data_dim,
                                hidden_dim=flags.num_hidden_unit,
                                mask=masks[i], num_layers=flags.num_hidden_layer)
                              for i in range(num_coupling_layers)])

    self.scaling_layer = ScalingLayer(data_dim=data_dim)

    self.prior = LogisticDistribution()

  def forward(self, x, invert=False):
    if not invert:
      z, log_det_jacobian = self.f(x)
      log_likelihood = torch.sum(self.prior.log_prob(z), dim=1) + log_det_jacobian
      return z, log_likelihood

    return self.f_inverse(x)

  def f(self, x):
    z = x
    log_det_jacobian = 0
    for i, coupling_layer in enumerate(self.coupling_layers):
      z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
    z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian)
    return z, log_det_jacobian

  def f_inverse(self, z):
    x = z
    x, _ = self.scaling_layer(x, 0, invert=True)
    for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
      x, _ = coupling_layer(x, 0, invert=True)
    return x

  def sample(self, num_samples):
    z = self.prior.sample([num_samples, self.data_dim]).view(self.samples, self.data_dim)
    return self.f_inverse(z)

  def _get_mask(self, dim, orientation=True):
    mask = np.zeros(dim)
    mask[::2] = 1.
    if orientation:
      mask = 1. - mask     # flip mask orientation
    mask = torch.tensor(mask)
    if cfg['USE_CUDA']:
      mask = mask.cuda()
    return mask.float()
