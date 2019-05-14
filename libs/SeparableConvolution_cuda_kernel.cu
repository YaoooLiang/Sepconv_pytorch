#include <stdio.h>

#include "SeparableConvolution_cuda_kernel.cuh"


#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>


#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))

#define DEBUG (0)
#ifndef BLOCKDIMX
#define BLOCKDIMX (32)
#endif
#ifndef BLOCKDIMY
#define BLOCKDIMY (16)
#endif
using at::Half;




//forward path of our layer
template <typename scalar_t>
__global__ void SeparableConvolution_gpu_forward_kernelfunc(
        const int nElement,
        const int w,        const int h,        const int channel, const int filter_size,

        const int input_b_stride, const int input_c_stride, const int input_h_stride, const int input_w_stride,
        const int vertical_b_stride, const int vertical_c_stride, const int vertical_h_stride, const int vertical_w_stride,
        const int horizontal_b_stride, const int horizontal_c_stride, const int horizontal_h_stride, const int horizontal_w_stride,
        const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

        const scalar_t* __restrict__    input,         const scalar_t* __restrict__    vertical,     const scalar_t* __restrict__    horizontal,     scalar_t*   output

        )
{

    //blockIdx.z : batch index from 0~B-1
    //blockIdx.y : height patch index from ceil(h/16)
    //blockIdx.x : width patch index from ceil(w/32)

    //threadidx.x: width index 0~31
    //threadIdx.y: height index 0~15
    //threadIdx.z: Not used

    //only use one dimensioon of the grid and block
    const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
    const bool withinXbounds = w_i < w;
    const bool withinYbounds = h_i < h;

    const int batch_i = blockIdx.z;
    const int off = batch_i * input_b_stride;

    if (withinXbounds && withinYbounds){
        for (int c_i = 0 ; c_i < channel ; c_i++){
            for (int intFilterY = 0; intFilterY < filter_size; intFilterY += 1){
                for (int intFilterX = 0; intFilterX < filter_size; intFilterX += 1){
                    atomicAdd(&output[batch_i * output_b_stride + c_i * output_c_stride + h_i * output_h_stride + w_i ], input[off + c_i * input_c_stride + (h_i + intFilterY) * input_h_stride + w_i + intFilterX] *
                                    vertical[batch_i * vertical_b_stride + intFilterY * vertical_c_stride + h_i * vertical_h_stride + w_i] * 
                                    horizontal[batch_i * horizontal_b_stride + intFilterX * horizontal_c_stride + h_i * horizontal_h_stride + w_i]);
                }
            }

        }

    }
    return ;

}


template <typename scalar_t>
__global__ void Vertical_gpu_backward_kernelfunc(
        const int nElement,        const int w,         const int h,        const int channel,  const int filter_size,
        const int input_b_stride, const int input_c_stride, const int input_h_stride, const int input_w_stride,
        const int horizontal_b_stride, const int horizontal_c_stride, const int horizontal_h_stride, const int horizontal_w_stride,
        const int vertical_b_stride, const int vertical_c_stride, const int vertical_h_stride, const int vertical_w_stride,
        const int gradoutput_b_stride, const int gradoutput_c_stride, const int gradoutput_h_stride, const int gradoutput_w_stride,
        const scalar_t* __restrict__      input,  const scalar_t* __restrict__    horizontal, const scalar_t* gradoutput,    scalar_t*  gradVertical
        )
        {
    //blockIdx.z : batch index from 0~B-1
    //blockIdx.y : height patch index from ceil(h/16)
    //blockIdx.x : width patch index from ceil(w/32)

    //threadidx.x: width index 0~31
    //threadIdx.y: height index 0~15
    //threadIdx.z: Not used

    const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
    const bool withinXbounds = w_i < w;
    const bool withinYbounds = h_i < h;

    const int batch_i = blockIdx.z;
    const int off  = batch_i * vertical_b_stride;

    if (withinXbounds && withinYbounds){
        for (int intFilterY = 0; intFilterY< filter_size; intFilterY += 1){
            for (int intFilterX = 0; intFilterX < filter_size; intFilterX += 1){
                for (int c_i = 0; c_i < channel; c_i ++){
                    atomicAdd(&gradVertical[off + intFilterY * vertical_c_stride + h_i * vertical_h_stride + w_i], gradoutput[batch_i * gradoutput_b_stride +
                        c_i * gradoutput_c_stride + h_i * gradoutput_w_stride + w_i] * input[batch_i * input_b_stride + c_i * input_c_stride + 
                        (h_i + intFilterY) * input_h_stride + intFilterX + w_i] * horizontal[batch_i * horizontal_b_stride + intFilterX * horizontal_c_stride + 
                        h_i * horizontal_h_stride + w_i]);

                }
            }
        }

        }
    return ;


}

template <typename scalar_t>
__global__ void Horizontal_gpu_backward_kernelfunc(
        const int nElement,        const int w,         const int h,        const int channel,  const int filter_size,
        const int input_b_stride, const int input_c_stride, const int input_h_stride, const int input_w_stride,
        const int horizontal_b_stride, const int horizontal_c_stride, const int horizontal_h_stride, const int horizontal_w_stride,
        const int vertical_b_stride, const int vertical_c_stride, const int vertical_h_stride, const int vertical_w_stride,
        const int gradoutput_b_stride, const int gradoutput_c_stride, const int gradoutput_h_stride, const int gradoutput_w_stride,
        const scalar_t* __restrict__      input,  const scalar_t* __restrict__    vertical, const scalar_t* gradoutput,    scalar_t*  gradHorizontal
        )
        {
    //blockIdx.z : batch index from 0~B-1
    //blockIdx.y : height patch index from ceil(h/16)
    //blockIdx.x : width patch index from ceil(w/32)

    //threadidx.x: width index 0~31
    //threadIdx.y: height index 0~15
    //threadIdx.z: Not used

    const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
    const bool withinXbounds = w_i < w;
    const bool withinYbounds = h_i < h;

    const int batch_i = blockIdx.z;
    const int off  = batch_i * vertical_b_stride;

    if (withinXbounds && withinYbounds){
        for (int intFilterX = 0; intFilterX< filter_size; intFilterX++){
            for (int intFilterY = 0; intFilterY < filter_size; intFilterY++){
                for (int c_i = 0; c_i < channel; c_i ++){
                    atomicAdd(&gradHorizontal[off + intFilterX * horizontal_c_stride + h_i * horizontal_h_stride + w_i], gradoutput[batch_i * gradoutput_b_stride +
                        c_i * gradoutput_c_stride + h_i * gradoutput_w_stride + w_i] * input[batch_i * input_b_stride + c_i * input_c_stride + 
                        (h_i + intFilterY) * input_h_stride + intFilterX + w_i] * vertical[batch_i * vertical_b_stride + intFilterY * vertical_c_stride + 
                        h_i * vertical_h_stride + w_i]);

                }
            }
        }

        }
    return ;


}

int SeparableConvolution_gpu_forward_kernel(
        cudaStream_t stream,
        const int nElement,
        const int w,        const int h,        const int channel,      const int batch, const  int filter_size,

        const int input_b_stride, const int input_c_stride, const int input_h_stride, const int input_w_stride,
        const int vertical_b_stride, const int vertical_c_stride, const int vertical_h_stride, const int vertical_w_stride,
        const int horizontal_b_stride, const int horizontal_c_stride, const int horizontal_h_stride, const int horizontal_w_stride,
        const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
        at::Tensor&  input,            at::Tensor&  vertical,        at::Tensor&  horizontal,    at::Tensor&  output

        )
{
    int error = 1 ;

    dim3 grid;
    dim3 block;


    //      blockthread = 128;
    //the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
    //the three channels are processsed in one kernel
    block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
    grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
    //extract the data of CudaTensor and use kernel to calculate.
        AT_DISPATCH_FLOATING_TYPES(input.type(), "SeparableConvolution_gpu_backward", ([&] {
SeparableConvolution_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
            nElement, //to let the nummous
            w,h,channel,filter_size,
            input_b_stride,input_c_stride,input_h_stride,input_w_stride,
            vertical_b_stride,vertical_c_stride,vertical_h_stride,vertical_w_stride,
            horizontal_b_stride,horizontal_c_stride,horizontal_h_stride,horizontal_w_stride,
            output_b_stride, output_c_stride, output_h_stride, output_w_stride,

            input.data<scalar_t>(),vertical.data<scalar_t>(),horizontal.data<scalar_t>(), output.data<scalar_t>()
            );
                    }));

    //          THCudaCheck(cudaGetLastError());
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("gpuerror in SeparableConver.updateOutput: %s\n", cudaGetErrorString(err));
        //THError("aborting");
        return error;
    }

    error = 0;
    return error;

}

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
        )
{

    int error = 1 ;

    dim3 grid;
    dim3 block;


    //blockthread = 128;
    //the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
    //the three channels are processsed in one kernel
    block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
    grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));
        AT_DISPATCH_FLOATING_TYPES(input.type(), "Vertical_gpu_backward", ([&] {
Vertical_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
            nElement, //to let the nummous
            w,h,channel,filter_size,
            input_b_stride,input_c_stride,input_h_stride,input_w_stride,
            horizontal_b_stride,horizontal_c_stride,horizontal_h_stride,horizontal_w_stride,
            vertical_b_stride,vertical_c_stride,vertical_h_stride,vertical_w_stride,
            gradoutput_b_stride, gradoutput_c_stride, gradoutput_h_stride, gradoutput_w_stride,


            input.data<scalar_t>(), horizontal.data<scalar_t>(), gradoutput.data<scalar_t>(),   gradVertical.data<scalar_t>()
            );
                    }));


        AT_DISPATCH_FLOATING_TYPES(input.type(), "Horizontal_gpu_backward", ([&] {
Horizontal_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
            nElement, //to let the nummous
            w,h,channel,filter_size,
            input_b_stride,input_c_stride,input_h_stride,input_w_stride,
            horizontal_b_stride,horizontal_c_stride,horizontal_h_stride,horizontal_w_stride,
            vertical_b_stride,vertical_c_stride,vertical_h_stride,vertical_w_stride,
            gradoutput_b_stride, gradoutput_c_stride, gradoutput_h_stride, gradoutput_w_stride,


            input.data<scalar_t>(), vertical.data<scalar_t>(), gradoutput.data<scalar_t>(),   gradHorizontal.data<scalar_t>()
            );
                    }));

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
        //THError("aborting");
        return error;
    }

    error = 0;
    return error;

}
