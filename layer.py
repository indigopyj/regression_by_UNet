import torch
import torch.nn as nn


# convolution 3X3, Batch norm, ReLU
class CBR2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()
        
        layers = []

        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)]
        if not norm is None:
            if norm == "bnorm":
              layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
              layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None:
            layers += [nn.ReLU() if relu == 0.0 else nn.LeakyRelu(relu)]

        self.cbr = nn.Sequential(*layers)

        

    def forward(self, x):
        return self.cbr(x)
