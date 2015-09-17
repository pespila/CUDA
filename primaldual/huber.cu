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

// uncomment to use the camera
// #define CAMERA

inline __device__ float l2Norm(float x1, float x2)
{
    return sqrtf(x1*x1 + x2*x2);
}

__global__ void init(float* xbar, float* xcur, float* xn, float* y1, float* y2, float* img, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int i = x + w * y;
    if (x < w && y < h)
    {
        float img_val = img[i];
        xbar[i] = img_val;
        xcur[i] = img_val;
        xn[i] = img_val;
        y1[i] = 0.f;
        y2[i] = 0.f;
    }
}

__global__ void prox_star(float* dx, float* dy, float* xbar, float sigma, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int i = x + w * y;
    int xi = (x+1) + w * y;
    int yi = x + w * (y+1);

    float d1, d2, norm, val;

    if (x < w && y < h)
    {
        val = xbar[i];
        norm = 0.f;
        d1 = dx[i] + sigma * (xbar[min(max(0, xi), w-1)]-val) * w;
        d2 = dy[i] + sigma * (xbar[min(max(0, yi), h-1)]-val) * h;
        norm = l2Norm(d1, d2);
        dx[i] = d1 / fmax(1.f, norm);
        dy[i] = d2 / fmax(1.f, norm);
    }
}

__global__ void prox_d(float* xn, float* dx, float* dy, float* xcur, float* img, float tau, float lambda, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int i = x + w * y;
    int xi = (x-1) + w * y;
    int yi = x + w * (y-1);
    

    if (x < w && y < h)
    {
        float d1, d2, val;
        d1 = (dx[i]-dx[min(max(0, xi), w-1)]) * w;
        d2 = (dy[i]-dy[min(max(0, yi), h-1)]) * h;
        val = xcur[i] + tau * (d1 + d2);
        xn[i] = (val + tau * lambda * img[i]) / (1.f + tau * lambda);
    }
}

__global__ void extrapolate(float* xbar, float* xn, float* xcur, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int i = x + w * y;

    if (x < w && y < h) {
        xbar[i] = 2.f * xn[i] - xcur[i];
    }
}

__global__ void set_xcur(float* xcur, float* xn, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int i = x + w * y;

    if (x < w && y < h) {
        xcur[i] = xn[i];
    }
}

__global__ void set_output(float* img, float* xbar, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int i = x + w * y;
    if (x < w && y < h)
    {
        img[i] = xbar[i];
    }
}

int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;

    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }

#endif
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;

    // number of computation repetitions to get a better run time measurement
    int dykstra = 1;
    getParam("dykstra", dykstra, argc, argv);
    cout << "dykstra: " << dykstra << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = true;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // load the input image as grayscale if "-gray" is specifed
    // float L = sqrtf(8) * h * w;
    // getParam("L", L, argc, argv);
    // cout << "L: " << L << endl;

    // load the input image as grayscale if "-gray" is specifed
    float tau = 0.01f;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    // float sigma = 1.f / (L*L*tau);
    // getParam("sigma", sigma, argc, argv);
    // cout << "sigma: " << sigma << endl;

    // load the input image as grayscale if "-gray" is specifed
    float lambda = 16.f;
    getParam("lambda", lambda, argc, argv);
    cout << "lambda: " << lambda << endl;

    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = 1;  // number of channels
    // int nc = mIn.channels();  // number of channels
    int size = w*h*nc;
    int nbytes = size*sizeof(float);
    float L2 = 8.f / (1.f / size);
    float sigma = 1.f / (L2*tau);
    cout << "image: " << w << " x " << h << endl;

    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    // allocate raw input image array
    float* h_imgIn  = new float[(size_t)size];
    float* h_imgOut = new float[(size_t)size];

    // allocate raw input image for GPU
    float* d_imgIn; cudaMalloc(&d_imgIn, nbytes); CUDA_CHECK;
    float* d_imgOut;cudaMalloc(&d_imgOut, nbytes); CUDA_CHECK;

    float* d_x; cudaMalloc(&d_x, nbytes); CUDA_CHECK;
    float* d_xbar; cudaMalloc(&d_xbar, nbytes); CUDA_CHECK;
    float* d_xcur; cudaMalloc(&d_xcur, nbytes); CUDA_CHECK;

    float* d_delX; cudaMalloc(&d_delX, nbytes); CUDA_CHECK;
    float* d_delY; cudaMalloc(&d_delY, nbytes); CUDA_CHECK;

    float* d_y1; cudaMalloc(&d_y1, nbytes); CUDA_CHECK;
    float* d_y2; cudaMalloc(&d_y2, nbytes); CUDA_CHECK;

    // size_t available, total;
    // cudaMemGetInfo(&available, &total);
    // cout << available << " " << total << endl;

    // alloc GPU memory

    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;

#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (h_imgIn, mIn);

    // copy host memory
    cudaMemcpy(d_imgIn, h_imgIn, nbytes, cudaMemcpyHostToDevice); CUDA_CHECK;

    // launch kernel
    // dim3 block = dim3(32, 8, nc);
    // dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1) / block.z);
    // dim3 block = dim3(32, 8, 4);
    // dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (level + block.z - 1) / block.z);
    dim3 block_iso = dim3(32, 8, 1);
    dim3 grid_iso = dim3((w + block_iso.x - 1) / block_iso.x, (h + block_iso.y - 1) / block_iso.y, 1);

    Timer timer; timer.start();
// void HuberROFModel::HuberROF(Image& src, WriteableImage& dst, Parameter& par) {
//     int i;
//     dst.Reset(height, width, src.GetType());
//     Initialize(src);
//     for (int k = 0; k < steps; k++)
//     {
//         for (i = 0; i < size; i++) {u_n[i] = u[i];}
//         Nabla(gradient_x, gradient_y, u_bar);
//         for (i = 0; i < size; i++) {gradient_x[i] = par.sigma * gradient_x[i] + p_x[i];}
//         for (i = 0; i < size; i++) {gradient_y[i] = par.sigma * gradient_y[i] + p_y[i];}
//         ProxRstar(p_x, p_y, gradient_x, gradient_y, par.alpha, par.sigma);
//         NablaTranspose(gradient_transpose, p_x, p_y);
//         for (i = 0; i < size; i++) {gradient_transpose[i] = u_n[i] - par.tau * gradient_transpose[i];}
//         ProxD(u, gradient_transpose, f, par.tau, par.lambda);
//         for (i = 0; i < size; i++) {u_bar[i] = u[i] + par.theta * (u[i] - u_n[i]);}
//     }
//     SetSolution(dst);
// }
    init <<<grid_iso, block_iso>>> (d_xbar, d_xcur, d_x, d_delX, d_delY, d_imgIn, w, h);
    for (int i = 0; i < repeats; i++)
    {
        set_xcur <<<grid_iso, block_iso>>> (d_xcur, d_x, w, h);
        prox_star <<<grid_iso, block_iso>>> (d_delX, d_delY, d_xbar, sigma, w, h);
        prox_d <<<grid_iso, block_iso>>> (d_x, d_delX, d_delY, d_xcur, d_imgIn, tau, lambda, w, h);
        extrapolate <<<grid_iso, block_iso>>> (d_xbar, d_x, d_xcur, w, h);
    }
    set_output <<<grid_iso, block_iso>>> (d_imgOut, d_xbar, w, h);

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    // cudaMemcpy(h_imgOut, d_imgIn, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(h_imgOut, d_imgOut, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;

    // free GPU memory
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    
    cudaFree(d_x); CUDA_CHECK;
    cudaFree(d_xbar); CUDA_CHECK;
    cudaFree(d_xcur); CUDA_CHECK;

    cudaFree(d_delX); CUDA_CHECK;
    cudaFree(d_delY); CUDA_CHECK;

    cudaFree(d_y1); CUDA_CHECK;
    cudaFree(d_y2); CUDA_CHECK;

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, h_imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif

    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] h_imgIn;
    delete[] h_imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}