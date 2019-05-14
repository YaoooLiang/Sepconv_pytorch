from torch.nn import Module
from .SeparableConvolutionLayer import SeparableConvolutionLayer


class SeparableConvolutionModule(Module):
    def __init__(self):
        super(SeparableConvolutionModule, self).__init__()

    def forward(self, input1, vertical, horizontal):
        return SeparableConvolutionLayer.apply(input1, vertical, horizontal)