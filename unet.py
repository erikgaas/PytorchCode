import torch
from torch import nn
import pretrainedmodels


def _grab_resnext_layers(net, parent_block_number, n_blocks):
    features = net.features
    names = [f'conv{parent_block_number}_{i}' for i in range(1, n_blocks+1)]
    return nn.Sequential(*[getattr(net, name) for name in names])

def _


class RNext101_UN(nn.Module):
    def __init__(self, ):
        super().__init__()

        dpn = pretrainedmodels.models.dpn107()

        conv1 = 



        dpn.features.conv1_1

        self.num_
        pass

    def 