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

__device__ float l2Norm(float x1, float x2)
{
    return sqrtf(x1*x1 + x2*x2);
}

__device__ float bound(float x1, float x2, float c)
{
    return 0.25f * (x1*x1 + x2*x2) - c;
}

__device__ void on_parabola(float* out, float x1, float x2, float x3, float c)
{
    float y = x3 + c;
    float norm = l2Norm(x1, x2);
    float v = 0.f;
    float a = 2.f * 0.25f * norm;
    float b = 2.f / 3.f * (1.f - 2.f * 0.25f * y);
    float d = b < 0 ? (a - pow(sqrt(-b), 3)) * (a + pow(sqrt(-b), 3)) : a*a + b*b*b;
    float ctmp = pow((a + sqrt(d)), 1.f/3.f);
    if (d >= 0) {
        v = ctmp == 0 ? 0.f : ctmp - b / ctmp;
    } else {
        v = 2.f * sqrt(-b) * cos((1.f / 3.f) * acos(a / (pow(sqrt(-b), 3))));
    }
    out[0] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x1 / norm;
    out[1] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x2 / norm;
    out[2] = bound(out[0], out[1], c);
}

__global__ void project_on_parabola(float* out, float* in, float c)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    float bound_val;
    float x1;
    float x2;
    float x3;

    if (x < 3)
    {
        x1 = in[0];
        x2 = in[1];
        x3 = in[2];
        bound_val = bound(x1, x2, c);

        if (x3 < bound_val) {
            on_parabola(out, x1, x2, x3, c);
        } else {
            out[0] = x1;
            out[1] = x2;
            out[2] = x3;
        }
    }
}

__device__ float partial(float x)
{
    return 0.5 * x;
}

__global__ void newton_parabola(float* out, float* in, float c)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    
    float bound_val;
    float x1, g1, p1, delta1, m11, tmp1;
    float x2, g2, p2, delta2, m22, tmp2;
    float x3, p3, m12, m21;

    if (x < 3)
    {
        p1 = in[0];
        p2 = in[1];
        p3 = in[3];
        x1 = p1;
        x2 = p2;
        x3 = p3;
        bound_val = bound(x1, x2, c);
        if (x3 < bound_val) {
            for (int k = 0; k < 100; k++)
            {
                g1 = x1 - p1 + partial(x1) * (bound(x1, x2, c) - p3);
                g2 = x2 - p2 + partial(x2) * (bound(x1, x2, c) - p3);

                m11 = 1.f + 0.5f * (bound(x1, x2, c) - p3) + (partial(x1) * partial(x1));
                m22 = 1.f + 0.5f * (bound(x1, x2, c) - p3) + (partial(x2) * partial(x2));
                m12 = partial(x1) * partial(x);
                m21 = m12;

                tmp1 = m11 == 0 ? 1.f : m11;
                tmp2 = m21 == 0 ? 1.f : m21;
                m11 *= tmp2; m12 *= tmp2; g1 *= tmp2;
                m21 *= tmp1; m22 *= tmp1; g2 *= tmp1;
                m21 -= m11;
                m22 -= m12;

                delta2 = m22 == 0 ? 0.f : -g2 / m22;
                delta1 = m11 == 0 ? 0.f : (-g1 - m12 * delta2) / m11;

                x1 += delta1;
                x2 += delta2;
            }
            out[0] = x1;
            out[1] = x2;
            out[2] = bound(x1, x2, c);
        } else {
            out[0] = x1;
            out[1] = x2;
            out[2] = x3;
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
    // load the input image as grayscale if "-gray" is specifed
    bool gray = true;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

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
    int size = 3;
    int nbyted = size*sizeof(float);
    cout << "image: " << w << " x " << h << endl;

    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    // allocate raw input image array
    float* h_in  = new float[(size_t)size];
    float* h_out = new float[(size_t)size];

    // allocate raw input image for GPU
    float* d_in; cudaMalloc(&d_in, nbyted); CUDA_CHECK;
    float* d_out;cudaMalloc(&d_out, nbyted); CUDA_CHECK;

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
    // convert_mat_to_layered (h_imgIn, mIn);

    // copy host memory
    h_in[0] = 21.f;
    h_in[1] = 9.f;
    h_in[2] = 9.f;

    // cout << "Vector = (" << h_in[0] << ", " << h_in[1] << ", " << h_in[2] << ")" << endl;
    printf("v = (%g, %g, %g)\n", h_in[0], h_in[1], h_in[2]);
    
    cudaMemcpy(d_in, h_in, nbyted, cudaMemcpyHostToDevice); CUDA_CHECK;

    // launch kernel
    dim3 block = dim3(128, 1, 1);
    dim3 grid = dim3((3 + block.x - 1) / block.x, 1, 1);

    Timer timer; timer.start();

    newton_parabola <<<grid, block>>> (d_out, d_in, 0.25f);
    cudaMemcpy(h_out, d_out, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;
    printf("Newton: v = (%g, %g, %g)\n", h_out[0], h_out[1], h_out[2]);
    // cout << "Newton: Vector = (" << h_out[0] << ", " << h_out[1] << ", " << h_out[2] << ")" << endl;
    
    project_on_parabola <<<grid, block>>> (d_out, d_in, 0.25f);
    cudaMemcpy(h_out, d_out, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;
    printf("Evgeny: v = (%g, %g, %g)\n", h_out[0], h_out[1], h_out[2]);
    // cout << "Evgeny: Vector = (" << h_out[0] << ", " << h_out[1] << ", " << h_out[2] << ")" << endl;

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;


    // free GPU memory
    cudaFree(d_in); CUDA_CHECK;
    cudaFree(d_out); CUDA_CHECK;

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

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

    // free allocated arrays
    delete[] h_in;
    delete[] h_out;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}