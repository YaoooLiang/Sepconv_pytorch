#include <torch/torch.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h> //works for 1.0.0

#include "SeparableConvolution_cuda_kernel.cuh"



int SeparableConvolution_gpu_forward(
        at::Tensor&  input,
        at::Tensor&  vertical,
        at::Tensor&  horizontal,
        at::Tensor&  output

        )
        {
    int error = 1 ;
    int channel = input.size(1);
    //if(channel!=3) return error;
    int batch = input.size(0);
    if(vertical.size(0) != batch) return error;
    if(vertical.size(1) != horizontal.size(1)) return error;

    int h = vertical.size(2);
    int w = vertical.size(3);
    if((input.size(2) - 51) != (h - 1)) return error;// to add some checkpoint
    if((input.size(3) - 51) != (w - 1)) return error;

    int filter_size = vertical.size(1);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));

    int input_b_stride = input.stride(0);
    int input_c_stride = input.stride(1);
    int input_h_stride = input.stride(2);
    int input_w_stride = input.stride(3);

    int vertical_b_stride = vertical.stride(0);
    int vertical_c_stride = vertical.stride(1);
    int vertical_h_stride = vertical.stride(2);
    int vertical_w_stride = vertical.stride(3);

    int horizontal_b_stride = horizontal.stride(0);
    int horizontal_c_stride = horizontal.stride(1);
    int horizontal_h_stride = horizontal.stride(2);
    int horizontal_w_stride = horizontal.stride(3);

    int output_b_stride = output.stride(0);
    int output_c_stride = output.stride(1);
    int output_h_stride = output.stride(2);
    int output_w_stride = output.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


    // TODO: do we need to assert the w_stride to be 1
    // printf("input_w is: %d, input_h: %d, input_c: %d, input_b: %d",  input_w_stride, input_h_stride, input_c_stride, input_w);
    // printf("output.stride(4) is: %d", output.stride(3));
    // printf("output.size(0) is: %d", output.size(0));
    // printf("output.size(1) is: %d", output.size(1));

    // if(input_b_stride != output.stride(0)) return error;
    // if(input_c_stride != output.stride(1)) return error;
    // printf("w: %d, h: %d, channel: %d, batch: %d, filter_size: %d\n", w, h, channel, batch, filter_size);

    int nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);

    error = SeparableConvolution_gpu_forward_kernel(
//          at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
            nElement,w,h,channel,batch, filter_size,

            input_b_stride,input_c_stride,input_h_stride,input_w_stride,
            vertical_b_stride,vertical_c_stride,vertical_h_stride,vertical_w_stride,
            horizontal_b_stride,horizontal_c_stride,horizontal_h_stride,horizontal_w_stride,
            output_b_stride,output_c_stride,output_h_stride,output_w_stride,


            input,
            vertical,
            horizontal,
            output);
      if (error) {AT_ERROR("CUDA call failed");}

    return error;

        }
int SeparableConvolution_gpu_backward(
        at::Tensor&  input,
        at::Tensor&  vertical,
        at::Tensor&  horizontal,
        at::Tensor&  gradoutput,
        at::Tensor&  gradinput,
        at::Tensor&  gradvertical,
        at::Tensor&  gradhorizontal
        )
        {


    int error = 1 ;
    int channel = input.size( 1);
    //if(channel!=3) return error;
    int batch = input.size(0);
    if(vertical.size(0) != batch) return error;
    if(vertical.size(1) != horizontal.size(1)) return error;

    int h = vertical.size(2);
    int w = vertical.size(3);
    if((input.size(2) - 51) != (h - 1)) return error;// to add some checkpoint
    if((input.size(3) - 51) != (w - 1)) return error;


    int filter_size = vertical.size(1);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));

    int input_b_stride = input.stride(0);
    int input_c_stride = input.stride(1);
    int input_h_stride = input.stride(2);
    int input_w_stride = input.stride(3);

    int vertical_b_stride = vertical.stride(0);
    int vertical_c_stride = vertical.stride(1);
    int vertical_h_stride = vertical.stride(2);
    int vertical_w_stride = vertical.stride(3);

    int horizontal_b_stride = horizontal.stride(0);
    int horizontal_c_stride = horizontal.stride(1);
    int horizontal_h_stride = horizontal.stride(2);
    int horizontal_w_stride = horizontal.stride(3);

    int gradoutput_b_stride = gradoutput.stride(0);
    int gradoutput_c_stride = gradoutput.stride(1);
    int gradoutput_h_stride = gradoutput.stride(2);
    int gradoutput_w_stride = gradoutput.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", horizontal_b_stride,horizontal_c_stride,horizontal_h_stride,horizontal_w_stride);


    if(input_b_stride != gradinput.stride(0)) return error;
    if(vertical_b_stride != gradvertical.stride(0)) return error;
    if(input_c_stride != gradinput.stride(1)) return error;
    if(vertical_c_stride != gradvertical.stride(1)) return error;
    if(horizontal_c_stride != gradhorizontal.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input_b_stride,input_c_stride,input_h_stride,input_w_stride);

    int nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

    error  = SeparableConvolution_gpu_backward_kernel(
//          at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
            nElement, //to let the nummous
            w,h,channel,batch, filter_size,

            input_b_stride,input_c_stride,input_h_stride,input_w_stride,
            horizontal_b_stride,horizontal_c_stride,horizontal_h_stride,horizontal_w_stride,
            vertical_b_stride,vertical_c_stride,vertical_h_stride,vertical_w_stride,
            gradoutput_b_stride,gradoutput_c_stride,gradoutput_h_stride,gradoutput_w_stride,

            input,
            vertical,
            horizontal,
            gradoutput,
            gradvertical,
            gradhorizontal
            );
      if (error) {AT_ERROR("CUDA call failed");}

    return error;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("SeparableConvolution_gpu_forward", &SeparableConvolution_gpu_forward, "SeparableConvolution forward (CUDA)");
  m.def("SeparableConvolution_gpu_backward", &SeparableConvolution_gpu_backward, "SeparableConvolution backward (CUDA)");
}
