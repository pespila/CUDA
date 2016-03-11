// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2015, September 7 - October 6
// ###
// ###
// ### Thomas Moellenhoff, Robert Maier, Caner Hazirbas
// ###
// ###
// ###
// ### THIS FILE IS SUPPOSED TO REMAIN UNCHANGED
// ###
// ###

#include "aux.h"
#include <iostream>
#include <stdio.h>
using namespace std;

__global__ void init(float* xbar, float* xcur, float* xn, float* y1, float* y2, float* img, int w, int h, int nc) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h) {
        int i;
        float val;

        for (int z = 0; z < nc; z++) {
            i = x + w * y + w * h * z;
            val = img[i];
            xbar[i] = val;
            xn[i] = val;
            xcur[i] = val;
            y1[i] = 0.f;
            y2[i] = 0.f;
        }
    }
}

__global__ void primal_descent(float* y1, float* y2, float* xbar, float sigma, float nu, float lambda, int w, int h, int nc) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h) {
        int i;
        float val, x1, x2, norm;
        float factor = (2.0 * lambda) / (sigma + 2.f * lambda);
        float bound = sqrtf((nu / lambda) * sigma * (sigma + 2.f * lambda));

        for (int z = 0; z < nc; z++) {
            i = x + w * y + w * h * z;

            val = xbar[i];
            x1 = (x+1<w) ? (xbar[(x+1) + w * y + w * h * z] - val) : 0.f;
            x2 = (y+1<h) ? (xbar[x + w * (y+1) + w * h * z] - val) : 0.f;

            x1 = y1[i] + sigma * x1;
            x2 = y2[i] + sigma * x2;

            norm = sqrtf(x1*x1+x2*x2);

            y1[i] = norm * factor <= bound ? factor * x1 : 0.f;
            y2[i] = norm * factor <= bound ? factor * x2 : 0.f;
        }
    }
}

__global__ void dual_ascent(float* xn, float* xcur, float* y1, float* y2, float* img, float tau, int w, int h, int nc) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h) {
        int i;
        float d1, d2, val;

        for (int z = 0; z < nc; z++) {
            i = x + w * y + w * h * z;

            d1 = y1[i] - (x>0 ? y1[(x-1) + w * y + w * h * z] : 0.f);
            d2 = y2[i] - (y>0 ? y2[x + w * (y-1) + w * h * z] : 0.f);
            val = xcur[i] + tau * (d1 + d2);
            xn[i] = (val + 2.f * tau * img[i]) / (1.f + 2.f * tau);
        }
    }
}

__global__ void extrapolate(float* xbar, float* xcur, float* xn, float theta, int w, int h, int nc) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h) {
        int i;

        for (int z = 0; z < nc; z++) {
            i = x + w * y + w * h * z;

            xbar[i] = xn[i] + theta * (xn[i] - xcur[i]);
            xcur[i] = xn[i];
        }
    }
}

__global__ void solution(float* img, float* xbar, int w, int h, int nc) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < w && y < h) {
        int i;

        for (int z = 0; z < nc; z++) {   
            i = x + w * y + w * h * z;

            img[i] = xbar[i];
        }
    }
}

int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;

    // Reading command line parameters:
    if (argc <= 2) { cout << "Usage: " << argv[0] << " -i <image> -o <output_image> [-repeats <repeats>] [-gray]" << endl; return 1; }
    
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    
    // output image
    string output = "";
    bool retO = getParam("o", output, argc, argv);
    if (!retO) cerr << "ERROR: no output image specified" << endl;

    // number of computation repetitions to get a better run time measurement
    int repeats = 140;
    getParam("repeats", repeats, argc, argv);
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    float tau = 0.25f;
    getParam("tau", tau, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    float sigma = 0.5f;
    getParam("sigma", sigma, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    float lambda = 2.f;
    getParam("lambda", lambda, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    float nu = 0.1f;
    getParam("nu", nu, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    float theta = 1.f;
    getParam("theta", theta, argc, argv);
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    int dim = w*h*nc;
    int nbyted = dim*sizeof(float);

    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    // allocate raw input image array
    float* h_img  = new float[(size_t)dim];

    // allocate raw input image for GPU
    float* d_imgInOut; cudaMalloc(&d_imgInOut, nbyted); CUDA_CHECK;

    float* d_x; cudaMalloc(&d_x, nbyted); CUDA_CHECK;
    float* d_xbar; cudaMalloc(&d_xbar, nbyted); CUDA_CHECK;
    float* d_xcur; cudaMalloc(&d_xcur, nbyted); CUDA_CHECK;

    float* d_y1; cudaMalloc(&d_y1, nbyted); CUDA_CHECK;
    float* d_y2; cudaMalloc(&d_y2, nbyted); CUDA_CHECK;

    // Init raw input image array
    convert_mat_to_layered (h_img, mIn);

    // copy host memory
    cudaMemcpy(d_imgInOut, h_img, nbyted, cudaMemcpyHostToDevice); CUDA_CHECK;

    // launch kernel
    dim3 block = dim3(32, 8, 1);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    Timer timer; timer.start();

    init <<<grid, block>>> (d_xbar, d_xcur, d_x, d_y1, d_y2, d_imgInOut, w, h, nc);
    for (int i = 1; i <= repeats; i++)
    {
        primal_descent <<<grid, block>>> (d_y1, d_y2, d_xbar, sigma, nu, lambda, w, h, nc);
        dual_ascent <<<grid, block>>> (d_x, d_xcur, d_y1, d_y2, d_imgInOut, tau, w, h, nc);
        theta = 1.f / sqrtf(1.f + 4.f * tau);
        tau *= theta;
        sigma /= theta;
        extrapolate <<<grid, block>>> (d_xbar, d_xcur, d_x, theta, w, h, nc);
    }
    solution <<<grid, block>>> (d_imgInOut, d_x, w, h, nc);

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "Time: " << t << " s" << endl;

    cudaMemcpy(h_img, d_imgInOut, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, h_img);
    showImage("Output", mOut, 100+w+40, 100);

    // wait for key inputs
    cv::waitKey(0);

    // save input and result
    cv::imwrite(output, mOut*255.f);

    // free GPU memory
    cudaFree(d_imgInOut); CUDA_CHECK;
    
    cudaFree(d_x); CUDA_CHECK;
    cudaFree(d_xbar); CUDA_CHECK;
    cudaFree(d_xcur); CUDA_CHECK;

    cudaFree(d_y1); CUDA_CHECK;
    cudaFree(d_y2); CUDA_CHECK;

    // free allocated arrays
    delete[] h_img;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}