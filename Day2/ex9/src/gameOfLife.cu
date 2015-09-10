#include <cuda_runtime.h>
#include <gameOfLife.h>
#include "aux.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void gameOfLifeKernel(unsigned char* d_src, unsigned char* d_dst, const size_t width, const size_t height) {

  /**
   *  YOUR CODE HERE
   *
   *  You must write here your kernel for one iteration of the game of life.
   *
   *  Input: d_src should contain the board at time 't'
   *  Output: d_dst should contain the board at time 't + 1' after one
   *  iteration of the game of life.
   *
   */

}

void runGameOfLifeIteration(unsigned char* d_src, unsigned char* d_dst, const size_t width, const size_t height) {
    
    // launch kernel
    dim3 block = dim3(width, height, 1);
    dim3 grid = dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

    gameOfLifeKernel <<<grid, block>>> (d_src, d_dst, width, height);
}
