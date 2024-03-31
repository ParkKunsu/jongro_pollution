import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
  """
    2-layer ANN model using time-series data
    Args:
      d_in: input dimension
      d_out: output_dimension
      c_in: input channel dimension, if use multi-input, set c > 1
      activation: activation layer for use
  """
  def __init__(self, d_in, d_out, d_hidden, c_in, activation=F.relu):
    super().__init__()
    self.lin1 = nn.Linear(d_in*c_in, d_hidden)
    self.lin2 = nn.Linear(d_hidden, d_out)
    self.activation = activation

  def forward(self, x):
    x = x.flatten(1)    # (B, d_in * c_in)
    x = self.lin1(x)    # (B, d_hidden)
    x = self.activation(x)
    x = self.lin2(x)    # (B, d_out)
    return F.sigmoid(x)