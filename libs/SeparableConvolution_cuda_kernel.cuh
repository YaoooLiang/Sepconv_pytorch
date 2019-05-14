#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

int SeparableConvolution_gpu_forward_kernel(
        cudaStream_t stream,
        const int nElement,
        const int w,        const int h,        const int channel,      const int batch, const  int filter_size,

        const int input_b_stride, const int input_c_stride, const int input_h_stride, const int input_w_stride,
        const int vertical_b_stride, const int vertical_c_stride, const int vertical_h_stride, const int vertical_w_stride,
        const int horizontal_b_stride, const int horizontal_c_stride, const int horizontal_h_stride, const int horizontal_w_stride,
        const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
        at::Tensor&  input,            at::Tensor&  vertical,        at::Tensor&  horizontal,    at::Tensor&  output

        );

int SeparableConvolution_gpu_backward_kernel(
        cudaStream_t stream,
        const int nElement,
        const int w,            const int h,            const int channel,          const int batch,            const int filter_size,

        const int input_b_stride, const int input_c_stride, const int input_h_stride, const int input_w_stride,
        const int horizontal_b_stride, const int horizontal_c_stride, const int horizontal_h_stride, const int horizontal_w_stride,
        const int vertical_b_stride, const int vertical_c_stride, const int vertical_h_stride, const int vertical_w_stride,
        const int gradoutput_b_stride, const int gradoutput_c_stride, const int gradoutput_h_stride, const int gradoutput_w_stride,

        at::Tensor&  input,            at::Tensor&  vertical,        at::Tensor&  horizontal,

        at::Tensor&  gradoutput,            at::Tensor&  gradVertical,        at::Tensor&  gradHorizontal
        );
