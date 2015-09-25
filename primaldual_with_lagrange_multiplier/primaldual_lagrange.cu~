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

__device__ float bound(float x1, float x2, float lambda, float k, float l, float f)
{
    return 0.25f * (x1*x1 + x2*x2) - lambda * pow(k / l - f, 2);
}

__device__ float interpolate(float k, float uk0, float uk1, float l)
{
    return (k + (0.5 - uk0) / (uk1 - uk0)) / l;
}

__device__ void on_parabola(float* u1, float* u2, float* u3, float x1, float x2, float x3, float f, float L, float lambda, float k, int j, float l)
{
    float y = x3 + lambda * pow(k / l - f, 2);
    float norm = l2Norm(x1, x2);
    float v = 0.f;
    float a = 2.f * 0.25f * norm;
    float b = 2.f / 3.f * (1.f - 2.f * 0.25f * y);
    float d = b < 0 ? (a - pow(sqrt(-b), 3)) * (a + pow(sqrt(-b), 3)) : a*a + b*b*b;
    float c = pow((a + sqrt(d)), 1.f/3.f);
    if (d >= 0) {
        v = c == 0 ? 0.f : c - b / c;
    } else {
        v = 2.f * sqrt(-b) * cos((1.f / 3.f) * acos(a / (pow(sqrt(-b), 3))));
    }
    u1[j] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x1 / norm;
    u2[j] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x2 / norm;
    u3[j] = bound(u1[j], u2[j], lambda, k, l, f);
}

__global__ void init(float* xbar, float* xcur, float* xn, float* y1, float* y2, float* y3, float* p1, float* p2, float* l1, float* l2, float* l1bar, float* l2bar, float* l1cur, float* l2cur, float* img, int w, int h, int l, int p)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h)
    {
        float img_val = img[x + w * y];
        for (int k = 0; k < p; k++)
        {
            int index = x + w * y + k * w * h;
            if (k < l) {
                xn[index] = img_val;
                xcur[index] = img_val;
                xbar[index] = img_val;
                y1[index] = 0.f;
                y2[index] = 0.f;
                y3[index] = 0.f;
            }
            p1[index] = 0.f;
            p2[index] = 0.f;
            l1[index] = 0.f;
            l2[index] = 0.f;
            l1cur[index] = 0.f;
            l2cur[index] = 0.f;
            l1bar[index] = 0.f;
            l2bar[index] = 0.f;
        }
    }
}

__global__ void parabola(float* y1, float* y2, float* y3, float* l1, float* l2, float* xbar, float* img, float sigma, float lambda, int k, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        float val = xbar[x + w * y + w * h * z];
        float x1 = (x+1<w) ? (xbar[(x+1) + w * y + w * h * z] - val) : 0.f;
        float x2 = (y+1<h) ? (xbar[x + w * (y+1) + w * h * z] - val) : 0.f;
        float x3 = (z+1<l) ? (xbar[x + w * y + w * h * (z+1)] - val) : 0.f;

        float lambda_sum = l1[x + w * y + w * h * k] + l2[x + w * y + w * h * k];
        x1 = y1[x + w * y + w * h * z] + sigma * (x1 + lambda_sum);
        x2 = y2[x + w * y + w * h * z] + sigma * (x2 + lambda_sum);
        x3 = y3[x + w * y + w * h * z] + sigma * x3;

        int index = x + w * y;

        float f = img[index];
        float bound_val = bound(x1, x2, lambda, (z+1.f), l, f);
        if (x3 < bound_val) {
            // on_parabola(y1, y2, y3, x1, x2, x3, f, 0.f, lambda, (z+1.f), x + w * y + w * h * z, l);
        } else {
            // y1[x + w * y + w * h * z] = x1;
            // y2[x + w * y + w * h * z] = x2;
            // y3[x + w * y + w * h * z] = x3;
        }
    }
}

__global__ void l2projection(float* p1, float* p2, float* l1bar, float* l2bar, float sigma, float nu, int k, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h)
    {
        float x1 = p1[x + w * y + w * h * k] + sigma * l1bar[x + w * y + w * h * k];
        float x2 = p2[x + w * y + w * h * k] + sigma * l2bar[x + w * y + w * h * k];

        float norm = l2Norm(x1, x2);
        // p1[x + w * y + w * h * k] = norm <= nu ? x1 : x1/norm * nu;
        // p2[x + w * y + w * h * k] = norm <= nu ? x2 : x2/norm * nu;
    }
}

__global__ void clipping(float* xn, float* xcur, float* y1, float* y2, float* y3, float tau, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        float d1 = y1[x + w * y + w * h * z] - (x>0 ? y1[(x-1) + w * y + w * h * z] : 0.f);
        float d2 = y2[x + w * y + w * h * z] - (y>0 ? y2[x + w * (y-1) + w * h * z] : 0.f);
        float d3 = y3[x + w * y + w * h * z] - (z>0 ? y3[x + w * y + w * h * (z-1)] : 0.f);
        float val = xcur[x + w * y + w * h * z] + tau * (d1 + d2 + d3);
        if (z == 0) {
            xn[x + w * y + w * h * z] = 1.f;
        } else if (z == l-1) {
            xn[x + w * y + w * h * z] = 0.f;
        } else {
            xn[x + w * y + w * h * z] = fmin(1.f, fmax(0.f, val));
        }
    }
}

__global__ void update_lambda(float* l1, float* l2, float* l1cur, float* l2cur, float* p1, float* p2, float* y1, float* y2, float tau, int k1, int k2, int K, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        float y1tmp = 0.f;
        float y2tmp = 0.f;
        for (int i = k1; i <= k2; i++)
        {
            // y1tmp += y1[x + w * y + w * h * i];
            // y2tmp += y2[x + w * y + w * h * i];       
        }
        // l1[x + w * y + w * h * K] = l1[x + w * y + w * h * K] - tau * (p1[x + w * y + w * h * K] - y1tmp);
        // l2[x + w * y + w * h * K] = l2[x + w * y + w * h * K] - tau * (p2[x + w * y + w * h * K] - y2tmp);
    }
}

__global__ void extrapolate(float* xbar, float* l1bar, float* l2bar, float* xcur, float* l1cur, float* l2cur, float* xn, float* l1, float* l2, int w, int h, int l, int p)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    int i = x + w * y + w * h * z;

    if (x < w && y < h && z < l) {
        xbar[i] = 2 * xn[i] - xcur[i];
        xcur[i] = xn[i];
        for (int k = 0; k < p; k++)
        {
            l1bar[x + w * y + w * h * k] = 2 * l1[x + w * y + w * h * k] - l1cur[x + w * y + w * h * k];
            l1cur[x + w * y + w * h * k] = l1[x + w * y + w * h * k];
            l2bar[x + w * y + w * h * k] = 2 * l2[x + w * y + w * h * k] - l2cur[x + w * y + w * h * k];
            l2cur[x + w * y + w * h * k] = l2[x + w * y + w * h * k];
        }
    }
}

__global__ void isosurface(float* img, float* xbar, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < w && y < h)
    {
        float val = 0.f;
        float uk0 = 0.f;
        float uk1 = 0.f;

        for (int k = 0; k < l-1; k++)
        {
            uk0 = xbar[x + w * y + k * w * h];
            uk1 = xbar[x + w * y + (k+1) * w * h];
            if (uk0 > 0.5 && uk1 <= 0.5) {
                val = interpolate(k+1, uk0, uk1, l);
                break;
            } else {
                val = 1.f;
            }
        }
        
        img[x + w * y] = val;
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
    bool gray = true;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // load the input image as grayscale if "-gray" is specifed
    int level = 16;
    getParam("level", level, argc, argv);
    cout << "level: " << level << endl;

    // load the input image as grayscale if "-gray" is specifed
    float taux = 1.f / 6.f;
    getParam("taux", taux, argc, argv);
    cout << "taux: " << taux << endl;

    // load the input image as grayscale if "-gray" is specifed
    float taul = 1.f;
    
    // load the input image as grayscale if "-gray" is specifed
    float sigmay = 1.f / 3.f;
    getParam("sigmay", sigmay, argc, argv);
    cout << "sigmay: " << sigmay << endl;

    // load the input image as grayscale if "-gray" is specifed
    float sigmap = 1.f;
    getParam("sigmap", sigmap, argc, argv);
    cout << "sigmap: " << sigmap << endl;

    // load the input image as grayscale if "-gray" is specifed
    float lambda = 1.f;
    getParam("lambda", lambda, argc, argv);
    cout << "lambda: " << lambda << endl;

    // load the input image as grayscale if "-gray" is specifed
    float nu = 5.f;
    getParam("nu", nu, argc, argv);
    cout << "nu: " << nu << endl;

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
    int nc = mIn.channels();  // number of channels
    int dim = w*h*nc;
    int size = w*h*nc*level;
    int proj = level * (level+1) / 2;
    int nbyted = dim*sizeof(float);
    int nbytes = size*sizeof(float);
    int nbytep = proj*dim*sizeof(float);
    cout << "image: " << w << " x " << h << endl;

    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    // allocate raw input image array
    float* h_imgIn  = new float[(size_t)dim];
    float* h_imgOut = new float[(size_t)dim];

    // allocate raw input image for GPU
    float* d_imgInOut; cudaMalloc(&d_imgInOut, nbyted); CUDA_CHECK;

    float* d_x; cudaMalloc(&d_x, nbytes); CUDA_CHECK;
    float* d_xbar; cudaMalloc(&d_xbar, nbytes); CUDA_CHECK;
    float* d_xcur; cudaMalloc(&d_xcur, nbytes); CUDA_CHECK;

    float* d_y1; cudaMalloc(&d_y1, nbytes); CUDA_CHECK;
    float* d_y2; cudaMalloc(&d_y2, nbytes); CUDA_CHECK;
    float* d_y3; cudaMalloc(&d_y3, nbytes); CUDA_CHECK;

    float* d_p1; cudaMalloc(&d_p1, nbytep); CUDA_CHECK;
    float* d_p2; cudaMalloc(&d_p2, nbytep); CUDA_CHECK;

    float* d_lambda1; cudaMalloc(&d_lambda1, nbytep); CUDA_CHECK;
    float* d_lambda2; cudaMalloc(&d_lambda2, nbytep); CUDA_CHECK;

    float* d_lambda1bar; cudaMalloc(&d_lambda1bar, nbytep); CUDA_CHECK;
    float* d_lambda2bar; cudaMalloc(&d_lambda2bar, nbytep); CUDA_CHECK;

    float* d_lambda1cur; cudaMalloc(&d_lambda1cur, nbytep); CUDA_CHECK;
    float* d_lambda2cur; cudaMalloc(&d_lambda2cur, nbytep); CUDA_CHECK;

    size_t available, total;
    cudaMemGetInfo(&available, &total);
    cout << "GPU Memory: " << total << " - " << available << " = " << total - available << endl;

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
    cudaMemcpy(d_imgInOut, h_imgIn, nbyted, cudaMemcpyHostToDevice); CUDA_CHECK;

    // launch kernel
    dim3 block = dim3(32, 8, 4);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (level + block.z - 1) / block.z);
    dim3 block_iso = dim3(32, 8, 1);
    dim3 grid_iso = dim3((w + block_iso.x - 1) / block_iso.x, (h + block_iso.y - 1) / block_iso.y, 1);

    Timer timer; timer.start();

    int K = 1;

    init <<<grid_iso, block_iso>>> (d_xbar, d_xcur, d_x, d_y1, d_y2, d_y3, d_p1, d_p2, d_lambda1, d_lambda2, d_lambda1bar, d_lambda2bar, d_lambda1cur, d_lambda2cur, d_imgInOut, w, h, level, proj);

    for (int i = 1; i <= repeats; i++)
    {
        for (int k1 = 0; k1 < level; k1++)
        {
            for (int k2 = k1; k2 < level; k2++)
            {
                parabola <<<grid, block>>> (d_y1, d_y2, d_y3, d_lambda1, d_lambda2, d_xbar, d_imgInOut, sigmay, lambda, K, w, h, level);
                l2projection <<<grid_iso, block_iso>>> (d_p1, d_p2, d_lambda1bar, d_lambda2bar, sigmap, nu, K, w, h);
                update_lambda <<<grid, block>>> (d_lambda1, d_lambda2, d_lambda1cur, d_lambda2cur, d_p1, d_p2, d_y1, d_y2, taul, k1, k2, K, w, h, level);
                K++;
            }
        }
        clipping <<<grid, block>>> (d_x, d_xcur, d_y1, d_y2, d_y3, taux, w, h, level);
        extrapolate <<<grid, block>>> (d_xbar, d_lambda1bar, d_lambda2bar, d_xcur, d_lambda1cur, d_lambda2cur, d_x, d_lambda1, d_lambda2, w, h, level, proj);
    }
    isosurface <<<grid_iso, block_iso>>> (d_imgInOut, d_x, w, h, level);

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    cudaMemcpy(h_imgOut, d_imgInOut, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;

    // free GPU memory
    cudaFree(d_imgInOut); CUDA_CHECK;
    
    cudaFree(d_x); CUDA_CHECK;
    cudaFree(d_xbar); CUDA_CHECK;
    cudaFree(d_xcur); CUDA_CHECK;

    cudaFree(d_y1); CUDA_CHECK;
    cudaFree(d_y2); CUDA_CHECK;
    cudaFree(d_y3); CUDA_CHECK;

    cudaFree(d_p1); CUDA_CHECK;
    cudaFree(d_p2); CUDA_CHECK;

    cudaFree(d_lambda1); CUDA_CHECK;
    cudaFree(d_lambda2); CUDA_CHECK;

    cudaFree(d_lambda1bar); CUDA_CHECK;
    cudaFree(d_lambda2bar); CUDA_CHECK;
    
    cudaFree(d_lambda1cur); CUDA_CHECK;
    cudaFree(d_lambda2cur); CUDA_CHECK;

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