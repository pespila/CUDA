#include <cuda_runtime.h>
#include <gameOfLife.h>
#include "aux.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void gameOfLifeKernel(unsigned char* d_src, unsigned char* d_dst, const size_t width, const size_t height) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int index = x + width * y;
    int left = x != 0 ? (x-1) + width * y : -1;
    int right = x < width-1 ? (x+1) + width * y : -1;
    int top = y != 0 ? x + width * (y-1) : -1;
    int bottom = y < height-1 ? x + width * (y+1) : -1;
    int status = d_src[index];
    int counter = 0;
    if (x < width && y < height) {
        if (top != -1) {
            counter += d_src[top] == 1 ? 1 : 0;
        }
        if (bottom != -1) {
            counter += d_src[bottom] == 1 ? 1 : 0;
        }
        if (left != -1) {
            counter += d_src[left] == 1 ? 1 : 0;
        }
        if (right != -1) {
            counter += d_src[right] == 1 ? 1 : 0;
        }

        if (status == 1) {
            if (counter < 2) {
                d_dst[index] = 0;
            } else if (counter > 3) {
                d_dst[index] = 0;
            } else {
                d_dst[index] = 1;
            }
        }
        if (status == 0) {
            if (counter == 3) {
                d_dst[index] = 1;
            }
        }
    }

}

void runGameOfLifeIteration(unsigned char* d_src, unsigned char* d_dst, const size_t width, const size_t height) {
    
    // launch kernel
    dim3 block = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid = dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

    gameOfLifeKernel <<<grid, block>>> (d_src, d_dst, width, height);
}
