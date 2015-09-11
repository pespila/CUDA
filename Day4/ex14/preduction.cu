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
#include <cublas_v2.h>
using namespace std;

// uncomment to use the camera
// #define CAMERA

__global__ void reduction(float* out, float* in, int size) {

    extern __shared__ float sh_data[];
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int th_id = threadIdx.x;

    float val = 0.f;
    if (x < size) {
        val = in[x];
    }
    sh_data[th_id] = val;
    __syncthreads();
    
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (th_id < offset) {
            sh_data[th_id] += sh_data[th_id + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sh_data[0];
    }

}

__global__ void sumup(float* out, float* in, int size) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < size) {
        for (int i = 0; i < size; i++)
        {
            out[0] += in[i];
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
    bool reduct = false;
    getParam("reduct", reduct, argc, argv);
    cout << "reduct: " << reduct << endl;

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
    // int nc = mIn.channels();  // number of channels
    int size = pow(10, 6);
    // int size = w * h * nc;
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
    float *h_imgIn  = new float[(size_t)size];
    float *h_imgOut = new float[(size_t)size];

    for (int i = 0; i < size; i++) {
        h_imgIn[i] = 1.f;
    }

    // allocate raw input image for GPU
    float* d_imgIn;
    float* d_imgOut;

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
    
    // convert_mat_to_layered (h_imgIn, mIn);

    // alloc GPU memory
    cudaMalloc(&d_imgIn, nbyte);
    CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbyte);
    CUDA_CHECK;

    // copy host memory
    cudaMemcpy(d_imgIn, h_imgIn, nbyte, cudaMemcpyHostToDevice);
    CUDA_CHECK;

    // launch kernel
    int b_size = 256;
    dim3 block = dim3(b_size, 1, 1);
    dim3 grid = dim3((size + block.x - 1) / block.x, 1, 1);
    size_t smBytes = block.x * block.y * block.z * sizeof(float);

    cublasStatus_t retrn;  
    cublasHandle_t handle;
    retrn = cublasCreate(&handle);

    int csize = 0;
    float sum = 0.f;

    Timer timer; timer.start();

    if (reduct) {
        reduction <<<grid, block, smBytes>>> (d_imgOut, d_imgIn, size);
        if (size%b_size == 0) {
            csize = size / b_size;
        } else {
            csize = size / b_size + 1;
        }
        sumup <<<grid, block>>> (&sum, d_imgOut, csize);
    } else {
        retrn = cublasSasum(handle, size, d_imgIn, 1, h_imgOut);
    }

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    cublasDestroy(handle);

    if (reduct) {
        cudaMemcpy(h_imgOut, d_imgOut, nbyte, cudaMemcpyDeviceToHost);
        CUDA_CHECK;
    }


    // float sum = 0.f;
    // for (int i = 0; i < csize; i++) {
    //     sum += h_imgOut[i];
    // }

    cout << "The sum is: " << sum << endl;

    // free GPU memory
    cudaFree(d_imgIn);
    CUDA_CHECK;
    cudaFree(d_imgOut);
    CUDA_CHECK;

    // show input image
    // showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    // convert_layered_to_mat(mOut, h_imgIn);
    // showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif

    // save input and result
    // cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    // cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] h_imgIn;
    delete[] h_imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}