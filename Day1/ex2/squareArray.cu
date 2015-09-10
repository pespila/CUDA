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

inline __device__ float square(float x) {
    return pow(x, 2);
}

__global__ void squareArray(float* in_out, int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size) {
        in_out[index] = square(in_out[index]);
    }
}

int main(int argc, char* argv[]) {
    int i;

    // define size of array by default
    int size = 24; // DEFAULT
    
    // alloc the arrays for the CPU
    float* h_value = new float[size];

    // define arrays for GPU
    float* d_value;
    unsigned long nbyte = size * sizeof(float);

    // define variable gauss_brace
    int gauss_brace = 0;

    // set values to each element of the arrays
    for (i = 0; i < size; i++) {
        h_value[i] = i;
    }

    // compute result on CPU
    for (i = 0; i < size; i++)
        h_value[i] = sqrtf(h_value[i]);

    // print the results and reset the h_sum
    cout << "Results of the CPU:" << endl;
    for (i = 0; i < size; i++) {
        cout << "Element " << i << ": " << h_value[i] << endl;
        h_value[i] = i;
    }

    // alloc GPU memory
    cudaMalloc(&d_value, nbyte);
    CUDA_CHECK;

    // copy host memory
    cudaMemcpy(d_value, h_value, nbyte, cudaMemcpyHostToDevice);
    CUDA_CHECK;

    // launch kernel
    dim3 block = dim3(32, 1, 1);
    gauss_brace = (size + block.x - 1) / block.x;
    dim3 grid = dim3(gauss_brace, 1, 1);
    squareArray <<<grid, block>>> (d_value, size);

    cudaMemcpy(h_value, d_value, nbyte, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // print the results again
    cout << "Results of the GPU:" << endl;
    for (i = 0; i < size; i++)
        cout << "Element " << i << ": " << h_value[i] << endl;

    // free CPU memory
    delete[] h_value;

    // free GPU memory
    cudaFree(d_value);
    CUDA_CHECK;
}