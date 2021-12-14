import math
from typing import final

import torch
import torch.nn as nn


class CvMNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape, self.output_shape = input_shape, output_shape

    @final
    def forward(self, x):
        # breakpoint()
        assert list(x.shape)[1:] == self.input_shape, list(x.shape)[1:]

        x = self._forward(x)
        assert list(x.shape)[1:] == self.output_shape, list(x.shape)[1:]
        return x

    def _forward(self, x):
        raise NotImplementedError


class TestNet(CvMNet):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        int_shape = [math.prod(input_shape)]
        self.cvmnet1 = CvmMnistNet(input_shape, int_shape)
        self.linnet2 = LinClassifier(int_shape, output_shape)

    def _forward(self, x):
        x = self.cvmnet1(x)
        return self.linnet2(x)


class LinClassifier(CvMNet):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        self.fc_in_dim = math.prod(input_shape)
        self.l1 = nn.Linear(self.fc_in_dim, math.prod(output_shape))

    def _forward(self, x):
        x = x.view(-1, self.fc_in_dim)
        assert list(x.shape)[1:] == [self.fc_in_dim], list(x.shape)[1:]
        return self.l1(x)


class CvmMnistNet(CvMNet):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)

        self.conv1 = nn.Conv2d(1, 1, (4, 4), )
        self.conv1_shape = get_output_shape_from_input_shape(self.conv1, self.input_shape)
        self.fc_in_dim = get_fc_dimensions_from_conv_output(self.conv1_shape)
        self.l1 = nn.Linear(self.fc_in_dim, math.prod(output_shape))
        # self.l1.

    def _forward(self, x):
        x = self.conv1(x)

        assert list(x.shape)[1:] == self.conv1_shape, [list(x.shape)[1:], self.conv1_shape]
        x = x.view(-1, self.fc_in_dim)
        assert list(x.shape)[1:] == [self.fc_in_dim], [list(x.shape)[1:], [self.fc_in_dim]]
        return self.l1(x)


class MLP(CvMNet):
    def __init__(self, input_shape, output_shape, activation_fn=nn.ReLU, normalization_layer=None, nr_intermediate_layers=None, width=None,
                 layerwise_width_list=None, no_activation=False):
        super().__init__(input_shape, output_shape)
        if layerwise_width_list is None:
            layerwise_width_list = [width for _ in range(0, nr_intermediate_layers)]
        else:
            assert nr_intermediate_layers is None and width is None
        self.fc_in_dim = math.prod(self.input_shape)

        layerwise_width_list.append(math.prod(self.output_shape))
        wi = self.fc_in_dim
        layers=[]
        for i,wo in enumerate(layerwise_width_list):
            layers.append(nn.Linear(wi, wo))
            wi = wo
            if activation_fn is not None and not i==len(layerwise_width_list)-1:
                layers.append(activation_fn())
            if normalization_layer is not None:
                layers.append(normalization_layer(num_features=wo))
            

        self.layers = nn.Sequential(*layers)

    def _forward(self, x):
        x = x.view(-1, self.fc_in_dim)
        return self.layers(x)


class FullyConnectedReLULayers(CvMNet):
    def __init__(self, input_shape, output_shape, width, nr_hidden):
        super().__init__(input_shape, output_shape)
        self.fc_in_dim = math.prod(self.input_shape)

        # layers = nn.ModuleList()
        layers = [nn.Linear(self.fc_in_dim, width), nn.ReLU()]
        for i in range(1, nr_hidden):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, math.prod(self.output_shape)))

        # self.fully_connected_relu_layers = layers
        self.fully_connected_relu_layers = nn.Sequential(*layers)
        print(self.fully_connected_relu_layers)
        # breakpoint()

    def _forward(self, x):
        # breakpoint()
        x = x.view(-1, self.fc_in_dim)
        return self.fully_connected_relu_layers(x)


# This shape doesn't include the batch-size.
def get_output_shape_from_input_shape(module, input_shape):
    if input_shape is None or len(input_shape) < 1:
        raise ValueError('the input shape needs to be non-empty')
    output_shape = list(module(torch.rand([1] + input_shape)).shape)
    le = len(output_shape)
    del output_shape[1]
    assert len(output_shape) == le - 1, {'output_shape': output_shape, 'le': le}

    return output_shape


def get_fc_dimensions_from_conv_output(conv_shape):
    fc_in_dim = math.prod(conv_shape)
    fc_in_dim2 = 1
    for i in conv_shape:
        fc_in_dim2 *= i
    assert fc_in_dim == fc_in_dim2  # just computing it twice to ensure no mistake
    return fc_in_dim
