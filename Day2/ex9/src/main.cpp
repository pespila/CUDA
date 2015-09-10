#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cuda_runtime.h>

#include <gameOfLife.h>
#include <BoardVisualization.hpp>

#include "aux.h"

#define BOARD_SIZE_X 200
#define BOARD_SIZE_Y 200
#define CIRCLE_RADIUS 2


void initBoardAtRandom(cv::Mat& board) {
    srand(time(NULL));
    for (int i = 0; i < board.rows; i++) {
        for (int j = 0; j < board.cols; j++) {
            board.at<unsigned char>(i, j) = static_cast<unsigned char>(rand() % 2);
        }
    }
}


int main(int argc, const char *argv[])
{
    unsigned char* d_src;

    cv::Mat board = cv::Mat::zeros(BOARD_SIZE_X, BOARD_SIZE_Y, CV_8UC1);
    BoardVisualization viewer(BOARD_SIZE_X, BOARD_SIZE_Y, CIRCLE_RADIUS);

    // Initialize the board randomly
    initBoardAtRandom(board);

    int nbyte = BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char);

    // pointer to the board array
    unsigned char* h_src = board.data;

    // alloc GPU memory
    cudaMalloc(&d_src, nbyte);
    CUDA_CHECK;

    cudaMemcpy(d_src, h_src, nbyte, cudaMemcpyHostToDevice);
    CUDA_CHECK;

    int key;
    while (key = cv::waitKey(10)) {

        runGameOfLifeIteration(d_src, d_src, BOARD_SIZE_X, BOARD_SIZE_Y);

        cudaMemcpy(h_src, d_src, nbyte, cudaMemcpyDeviceToHost);
        CUDA_CHECK;

        for (int i = 0; i < BOARD_SIZE_Y; i++)
        {
            for (int j = 0; j < BOARD_SIZE_X; j++)
            {
                board.at<uchar>(i, j) = h_src[j + i * BOARD_SIZE_X];
            }
        }

        /** This is just for display. You should not touch this.  **/
        viewer.displayBoard(board);
        if (key != -1) break;
    }

    cudaFree(d_src);
    CUDA_CHECK;
    free(h_src);

    return 0;
}
