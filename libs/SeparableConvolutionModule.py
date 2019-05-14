from torch.nn import Module
from .SeparableConvolutionLayer import SeparableConvolutionLayer


class SeparableConvolutionModule(Module):
    def __init__(self):
        super(SeparableConvolutionModule, self).__init__()
        # self.f = FilterInterpolationLayer()

    def forward(self, input, vertical, horizontal):
        return SeparableConvolutionLayer.apply(input, vertical, horizontal)