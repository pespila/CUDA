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

#define PI 3.14159265359

// uncomment to use the camera
// #define CAMERA

__global__ void fillG(float* G, float* src, float eps, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.z + blockDim.z * blockIdx.z;

    int index = x + width * y + width * height * c;
    int index_x = x+1 + width * y + width * height * c;
    int index_y = x + width * (y+1) + width * height * c;

    float delx, dely, l2;

    if (x < width && y < height) {
        delx = x < width + 1 ? src[index_x] - src[index] : 0.f;
        dely = y < height + 1 ? src[index_y] - src[index] : 0.f;

        l2 = sqrtf(delx*delx + dely*dely);
        G[index] = 1.f / fmax(eps, l2);
    }
}

__global__ void make_update(float* out, float* in, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.z + blockDim.z * blockIdx.z;

    int index = x + width * y + width * height * c;
    if (x < width && y < height) {
        out[index] = in[index];
    }
}

__global__ void jacobi(float* out, float* in, float* f, float* G, float lambda, float eps, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.z + blockDim.z * blockIdx.z;

    int index = x + width * y + width * height * c;
    int indexXM = (x-1) + width * y + width * height * c;
    int indexXP = (x+1) + width * y + width * height * c;
    int indexYM = x + width * (y-1) + width * height * c;
    int indexYP = x + width * (y+1) + width * height * c;

    if (x < width && y < height) {
        float g_r = x+1 < width ? G[indexXP] : 0.f;
        float u_r = x+1 < width ? in[indexXP] : 0.f;

        float g_u = y+1 < height ? G[indexYP] : 0.f;
        float u_u = y+1 < height ? in[indexYP] : 0.f;

        float g_l = x > 0 ? G[indexXM] : 0.f;
        float u_l = x > 0 ? in[indexXM] : 0.f;

        float g_d = y > 0 ? G[indexYM] : 0.f;
        float u_d = y > 0 ? in[indexYM] : 0.f;

        float denom = 2.f + lambda * (g_r + g_u + g_l + g_d);

        out[index] = (2.f * f[index] + lambda * (g_r * u_r + g_l * u_l + g_u * u_u + g_d * u_d)) / denom;
    }
}

__global__ void sor_method(float* out, float* in, float* f, float* G, float lambda, float theta, float eps, int rb, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.z + blockDim.z * blockIdx.z;

    x *= 2;
    if (y%2 != rb) x++;

    int index = x + width * y + width * height * c;
    int indexXM = (x-1) + width * y + width * height * c;
    int indexXP = (x+1) + width * y + width * height * c;
    int indexYM = x + width * (y-1) + width * height * c;
    int indexYP = x + width * (y+1) + width * height * c;

    float zk = 0.f;
    float z = in[index];
    float img = f[index];

    if (x < width && y < height) {
        float g_r = x+1 < width ? G[indexXP] : 0.f;
        float u_r = x+1 < width ? in[indexXP] : 0.f;

        float g_u = y+1 < height ? G[indexYP] : 0.f;
        float u_u = y+1 < height ? in[indexYP] : 0.f;

        float g_l = x > 0 ? G[indexXM] : 0.f;
        float u_l = x > 0 ? in[indexXM] : 0.f;

        float g_d = y > 0 ? G[indexYM] : 0.f;
        float u_d = y > 0 ? in[indexYM] : 0.f;

        float denom = 2.f + lambda * (g_r + g_u + g_l + g_d);

        zk = (2.f * img + lambda * (g_r * u_r + g_l * u_l + g_u * u_u + g_d * u_d)) / denom;
        out[index] = zk + theta * (zk - z);
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
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // load the input image as grayscale if "-gray" is specifed
    bool sor = false;
    getParam("sor", sor, argc, argv);
    cout << "sor: " << sor << endl;

    // load the input image as grayscale if "-gray" is specifed
    float lambda = 8.0;
    getParam("lambda", lambda, argc, argv);
    cout << "lambda: " << lambda << endl;

    // load the input image as grayscale if "-gray" is specifed
    float eps = 0.01;
    getParam("eps", eps, argc, argv);
    cout << "eps: " << eps << endl;

    // load the input image as grayscale if "-gray" is specifed
    float theta = 0.5;
    getParam("theta", theta, argc, argv);
    cout << "theta: " << theta << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    // float sigma = sqrtf(2.f * tau * repeats);
    // getParam("sigma", sigma, argc, argv);
    // cout << "sigma: " << sigma << endl;
    // int radius = ceil(3 * sigma);
    // int diameter = 2 * radius + 1;

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
    addNoise(mIn, 0.1);
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    int size = w * h * nc;
    int nbyte = size * sizeof(float);
    cout << "image: " << w << " x " << h << endl;

    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed

    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *h_imgIn  = new float[(size_t)size];
    float *h_imgOut = new float[(size_t)w*h*mOut.channels()];

    // allocate raw input image for GPU
    float* d_imgIn;
    float* d_f;
    float* d_imgOut;
    float* d_delX;
    float* d_delY;
    float* d_G;
    float* d_sor;

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

    // alloc GPU memory
    cudaMalloc(&d_imgIn, nbyte);
    CUDA_CHECK;
    cudaMalloc(&d_f, nbyte);
    CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbyte);
    CUDA_CHECK;
    cudaMalloc(&d_delX, nbyte);
    CUDA_CHECK;
    cudaMalloc(&d_delY, nbyte);
    CUDA_CHECK;
    cudaMalloc(&d_G, nbyte);
    CUDA_CHECK;
    cudaMalloc(&d_sor, nbyte);
    CUDA_CHECK;

    cudaMemcpy(d_imgIn, h_imgIn, nbyte, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMemcpy(d_f, h_imgIn, nbyte, cudaMemcpyHostToDevice);
    CUDA_CHECK;

    // launch kernel
    dim3 block = dim3(32, 8, nc);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1) / block.z);
    // dim3 block_sum_up = dim3(256, 1, 1);
    // dim3 grid_sum_up = dim3((size + block_sum_up.x - 1) / block_sum_up.x, 1, 1);
    // dim3 block_matrix = dim3(32, 8, 1);
    // dim3 grid_matrix = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    Timer timer; timer.start();
    for (int i = 1; i <= repeats; i++) {
        // del_x_plus <<<grid, block>>> (d_delX, d_imgIn, w, h);
        // del_y_plus <<<grid, block>>> (d_delY, d_imgIn, w, h);
        fillG <<<grid, block>>> (d_G, d_imgIn, eps, w, h);
        if (sor) {
            sor_method <<<grid, block>>> (d_imgIn, d_imgIn, d_f, d_G, lambda, theta, eps, 0, w, h);
            sor_method <<<grid, block>>> (d_imgIn, d_imgIn, d_f, d_G, lambda, theta, eps, 1, w, h);
        } else {
            jacobi <<<grid, block>>> (d_imgIn, d_imgIn, d_f, d_G, lambda, eps, w, h);
        }
        // if (sor) {
        //     if (i == repeats) {
        //         sor_method <<<grid, block>>> (d_imgOut, d_imgIn, d_f, d_G, lambda, theta, eps, 0, w, h);
        //         sor_method <<<grid, block>>> (d_imgOut, d_imgIn, d_f, d_G, lambda, theta, eps, 1, w, h);
        //     } else {
        //         sor_method <<<grid, block>>> (d_imgIn, d_imgIn, d_f, d_G, lambda, theta, eps, 0, w, h);
        //         sor_method <<<grid, block>>> (d_imgIn, d_imgIn, d_f, d_G, lambda, theta, eps, 1, w, h);
        //     }
        // } else {
        //     if (i == repeats) {
        //         jacobi <<<grid, block>>> (d_imgOut, d_imgIn, d_f, d_G, lambda, eps, w, h);
        //     } else {
        //         jacobi <<<grid, block>>> (d_imgIn, d_imgIn, d_f, d_G, lambda, eps, w, h);
        //     }
        // }
        // apply_g <<<grid, block>>> (d_delX, d_delY, w, h, kind, eps);
        // del_x_minus <<<grid, block>>> (d_divX, d_delX, w, h);
        // del_y_minus <<<grid, block>>> (d_divY, d_delY, w, h);
        // addArray <<<grid_sum_up, block_sum_up>>> (d_divergence, d_divX, d_divY, size);
        // if (i == repeats) {
        //     make_update <<<grid_matrix, block_matrix>>> (d_imgOut, d_imgIn, d_divergence, tau, eps, kind, w, h, nc);
        // } else {
        //     make_update <<<grid_matrix, block_matrix>>> (d_imgIn, d_imgIn, d_divergence, tau, eps, kind, w, h, nc);
        // }
    }

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    cudaMemcpy(h_imgOut, d_imgIn, nbyte, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free GPU memory
    cudaFree(d_imgIn);
    CUDA_CHECK;
    cudaFree(d_f);
    CUDA_CHECK;
    cudaFree(d_imgOut);
    CUDA_CHECK;
    cudaFree(d_delX);
    CUDA_CHECK;
    cudaFree(d_delY);
    CUDA_CHECK;
    cudaFree(d_G);
    CUDA_CHECK;
    cudaFree(d_sor);
    CUDA_CHECK;

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, h_imgOut);
    showImage("Diffusion", mOut, 100+w+40, 100);

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
    cv::imwrite("image_result.png",mOut*255.f);  // "imwrite" assumes channel range [0,255]

    // free allocated arrays
    delete[] h_imgIn;
    delete[] h_imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}