import torch
from torch.autograd import Function
import SeparableConvolution_cuda as my_lib


class SeparableConvolutionLayer(Function):
    """docstring for SeparableConvolutionLayer"""
    def __init__(self):
        super(SeparableConvolutionLayer, self).__init__()

    def forward(ctx, input, vertical, horizontal):
        assert(input.is_contiguous())
        assert(vertical.is_contiguous())
        assert(horizontal.is_contiguous())

        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert(intInputHeight - 51 == intOutputHeight - 1)
        assert(intInputWidth - 51 == intOutputWidth - 1)
        assert(intFilterSize == 51)

        output = torch.cuda.FloatTensor().resize_(intBatches, intInputDepth, intOutputHeight, intOutputWidth).zero_()
        # print(input.size())
        # print(vertical.size())
        # print(horizontal.size())
        # print(output.size())
        error = my_lib.SeparableConvolution_gpu_forward(input, vertical, horizontal, output)

        ctx.save_for_backward(input, vertical, horizontal)
        print(error)

        return output

    def backward(ctx, gradoutput):

        input, vertical, horizontal = ctx.saved_tensors

        gradinput = torch.cuda.FloatTensor().resize_(input.size()).zero_()
        gradvertical = torch.cuda.FloatTensor().resize_(vertical.size()).zero_()
        gradhorizontal = torch.cuda.FloatTensor().resize_(horizontal.size()).zero_()

        err = my_lib.SeparableConvolution_gpu_backward(input, vertical, horizontal, gradoutput, gradinput, gradvertical, gradhorizontal)
        if err != 0:
            print(err)

        return gradinput, gradvertical, gradhorizontal





