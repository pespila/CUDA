// Exercise 2 "First CUDA Kernels" of the first day's exercise sheet

// Load libraries
#include <cuda_runtime.h>
#include <iostream>

// use namespaces
using namespace std;

// cuda error checking (function provided by TUM, Thomas Moellenhoff, Robert Maier, Caner Hazirbas)
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}

inline __device__ float add(float x1, float x2) {
    return x1 + x2;
}

__global__ void addArray(float* out, float* in1, float* in2, int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size) {
        out[index] = add(in1[index], in2[index]);
    }
}

int main(int argc, char* argv[]) {
    int i;

    // define size of array by default
    int size = 24; // DEFAULT
    
    // alloc the arrays for the CPU
    float* h_sum = new float[size];
    float* h_x1 = new float[size];
    float* h_x2 = new float[size];

    // define arrays for GPU
    float* d_sum;
    float* d_x1;
    float* d_x2;
    unsigned long nbyte = size * sizeof(float);

    // define variable gauss_brace
    int gauss_brace = 0;

    // set values to each element of the arrays
    for (i = 0; i < size; i++) {
        h_sum[i] = 0;
        h_x1[i] = i;
        h_x2[i] = (i%5) + 1;
    }

    // compute result on CPU
    for (i = 0; i < size; i++)
        h_sum[i] = h_x1[i] + h_x2[i];

    // print the results and reset the h_sum
    cout << "Results of the CPU:" << endl;
    for (i = 0; i < size; i++) {
        cout << "Element " << i << ": " << h_x1[i] << " + " << h_x2[i] << " = " << h_sum[i] << endl;
        h_sum[i] = 0;
    }

    // alloc GPU memory
    cudaMalloc(&d_sum, nbyte);
    CUDA_CHECK;
    cudaMalloc(&d_x1, nbyte);
    CUDA_CHECK;
    cudaMalloc(&d_x2, nbyte);
    CUDA_CHECK;

    // copy host memory
    cudaMemcpy(d_sum, h_sum, nbyte, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMemcpy(d_x1, h_x1, nbyte, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMemcpy(d_x2, h_x2, nbyte, cudaMemcpyHostToDevice);
    CUDA_CHECK;

    // launch kernel
    dim3 block = dim3(32, 1, 1);
    gauss_brace = (size + block.x - 1) / block.x;
    dim3 grid = dim3(gauss_brace, 1, 1);
    addArray <<<grid, block>>> (d_sum, d_x1, d_x2, size);

    cudaMemcpy(h_sum, d_sum, nbyte, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // print the results again
    cout << "Results of the GPU:" << endl;
    for (i = 0; i < size; i++)
        cout << "Element " << i << ": " << h_x1[i] << " + " << h_x2[i] << " = " << h_sum[i] << endl;

    // free CPU memory
    delete[] h_x1;
    delete[] h_x2;
    delete[] h_sum;

    // free GPU memory
    cudaFree(d_x1);
    CUDA_CHECK;
    cudaFree(d_x2);
    CUDA_CHECK;
    cudaFree(d_sum);
    CUDA_CHECK;
}