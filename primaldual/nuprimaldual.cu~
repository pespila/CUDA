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

void Print1D(float* x, int w, int h, int l) {
    for (int k = 0; k < l; k++) {
        cout << "_____ Level: " << k << " _____" << endl;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cout << "( ";
                // for (int c = 0; c < Dimension(); c++) {
                    // printf("%g ", Get(i, j, k, c));
                    cout << x[j + i * w + k * h * w] << " ";
                // }
                cout << ")" << "  ";
            }
            cout << endl;
        }
        cout << "_____          _____" << endl << endl;
    }
}

void Print3D(float* x1, float* x2, float* x3, int w, int h, int l) {
    for (int k = 0; k < l; k++) {
        cout << "_____ Level: " << k << " _____" << endl;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cout << "( ";
                cout << x1[j + i * w + k * h * w] << " " << x2[j + i * w + k * h * w] << " " << x3[j + i * w + k * h * w] << " ";
                cout << ")" << "  ";
            }
            cout << endl;
        }
        cout << "_____          _____" << endl << endl;
    }
}

__device__ float l2Norm(float x1, float x2)
{
    return sqrtf(x1*x1 + x2*x2);
}

__device__ float bound(float x1, float x2, float lambda, float k, float L, float f)
{
    return 0.25f * (x1*x1 + x2*x2) - lambda * pow(k / L - f, 2);
}

__device__ float interpolate(float k, float uk0, float uk1, float l)
{
    return (k/l + (0.5 - uk0) / (uk1 - uk0));
}

__device__ void on_parabola(float* u1, float* u2, float* u3, float x1, float x2, float x3, float f, float L, float lambda, float k, int i)
{
    float y = x3 + lambda * pow(k / L - f, 2);
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
    u1[i] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x1 / norm;
    u2[i] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x2 / norm;
    u3[i] = bound(u1[i], u2[i], lambda, k, L, f);
}

__global__ void project_on_parabola(float* u1, float* u2, float* u3, float* tmp1, float* tmp2, float* tmp3, float* img, float L, float lambda, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        float f = img[x + w * y];
        float x1 = tmp1[x + w * y + w * h * z];
        float x2 = tmp2[x + w * y + w * h * z];
        float x3 = tmp3[x + w * y + w * h * z];
        float bound_val = bound(x1, x2, lambda, z, L, f);

        if (x3 < bound_val) {
            on_parabola(u1, u2, u3, x1, x2, x3, f, L, lambda, z, x + w * y + w * h * z);
        } else {
            u1[x + w * y + w * h * z] = x1;
            u2[x + w * y + w * h * z] = x2;
            u3[x + w * y + w * h * z] = x3;
        }
    }
}

__device__ float partial(float x)
{
    return 0.5 * x;
}

__global__ void soft_shrinkage(float* u1, float* u2, float* u3, float* tmp1, float* tmp2, float* tmp3, float nu, int k1, int k2, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    
    const float K = (float)(k2 - k1 + 1);

    if (x < w && y < h)
    {
        int i;
        float s1 = 0.f;
        float s2 = 0.f;
        float s01 = 0.f;
        float s02 = 0.f;
        float x1 = 0.f;
        float x2 = 0.f;

        for (int k = k1; k <= k2; k++)
        {
            i = x + w * y + k * w * h;
            x1 = tmp1[i];
            x2 = tmp2[i];
            s01 += x1;
            s02 += x2;
        }

        float norm = l2Norm(s01, s02);

        s1 = norm <= nu ? s01 : nu * s01 / norm;
        s2 = norm <= nu ? s02 : nu * s02 / norm;

        for (int k = 0; k < l; k++)
        {
            i = x + w * y + k * w * h;
            x1 = tmp1[i];
            x2 = tmp2[i];
            if (k >= k1 && k <= k2) {
                u1[i] = x1 + (s1 - s01) / K;
                u2[i] = x2 + (s2 - s02) / K;
            } else {
                u1[i] = x1;
                u2[i] = x2;
            }
        }
    }
}

__global__ void init(float* xbar, float* xcur, float* xn, float* y1, float* y2, float* y3, float* img, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h)
    {
        int index;
        float img_val = img[x + w * y];
        for (int k = 0; k < l; k++)
        {
            index = x + w * y + k * w * h;
            xn[index] = img_val;
            xcur[index] = img_val;
            xbar[index] = img_val;
            y1[index] = 0.f;
            y2[index] = 0.f;
            y3[index] = 0.f;
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
                val = interpolate(k, uk0, uk1, l);
                break;
            } else {
                val = 1.f;
            }
        }
        
        img[x + w * y] = val;
    }
}

__global__ void set_y(float* y1, float* y2, float* y3, float* u1, float* u2, float* u3, int w, int h, int l, int p)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x < w && y < h && z < l)
    {
        y1[x + w * y + w * h * z] = u1[x + w * y + w * h * z];
        y2[x + w * y + w * h * z] = u2[x + w * y + w * h * z];
        y3[x + w * y + w * h * z] = u3[x + w * y + w * h * z];
    }
}

__global__ void set_u_v(float* u1, float* u2, float* u3, float* v1, float* v2, float* v3, float* dx, float* dy, float* dz, int w, int h, int l, int p)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        int j;
        u1[x + w * y + w * h * z] = dx[x + w * y + w * h * z];
        u2[x + w * y + w * h * z] = dy[x + w * y + w * h * z];
        u3[x + w * y + w * h * z] = dz[x + w * y + w * h * z];
        
        for (int k = 0; k < p; k++)
        {
            j = x + w * y + w * h * z + k * w * h * l;
            v1[j] = 0.f;
            v2[j] = 0.f;
            v3[j] = 0.f;
            
        }
    }
}

__global__ void update_v(float* v1, float* v2, float* v3, float* u1, float* u2, float* u3, float* tmp1, float* tmp2, float* tmp3, int w, int h, int l, int k)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    

    if (x < w && y < h && z < l)
    {
        v1[x + w * y + w * h * z + k * w * h * l] = u1[x + w * y + w * h * z] - (tmp1[x + w * y + w * h * z]);
        v2[x + w * y + w * h * z + k * w * h * l] = u2[x + w * y + w * h * z] - (tmp2[x + w * y + w * h * z]);
        v3[x + w * y + w * h * z + k * w * h * l] = u3[x + w * y + w * h * z] - (tmp3[x + w * y + w * h * z]);
    }
}

__global__ void set_tmp_u(float* tmp1, float* tmp2, float* tmp3, float* u1, float* u2, float* u3, float* v1, float* v2, float* v3, int w, int h, int l, int p)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x < w && y < h && z < l)
    {
        tmp1[x + w * y + w * h * z] = u1[x + w * y + w * h * z] - v1[x + w * y + w * h * z + p * w * h * l];
        tmp2[x + w * y + w * h * z] = u2[x + w * y + w * h * z] - v2[x + w * y + w * h * z + p * w * h * l];
        tmp3[x + w * y + w * h * z] = u3[x + w * y + w * h * z] - v3[x + w * y + w * h * z + p * w * h * l];
    }
}

__global__ void calc_norm(float* norm, float* r1, float* r2, float* r3, float* u1, float* u2, float* u3, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        float a = pow(r1[x + w * y + w * h * z] - u1[x + w * y + w * h * z], 2);
        float b = pow(r2[x + w * y + w * h * z] - u2[x + w * y + w * h * z], 2);
        float c = pow(r3[x + w * y + w * h * z] - u3[x + w * y + w * h * z], 2);
        norm[x + w * y + w * h * z] = a+b+c;
    }
}

__global__ void set_r(float* r1, float* r2, float* r3, float* u1, float* u2, float* u3, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x < w && y < h && z < l)
    {
        r1[x + w * y + w * h * z] = u1[x + w * y + w * h * z];
        r2[x + w * y + w * h * z] = u2[x + w * y + w * h * z];
        r3[x + w * y + w * h * z] = u3[x + w * y + w * h * z];
    }
}

// int i = x + w * y + w * h * z;
//     int xi = (x+1) + w * y + w * h * z;
//     int yi = x + w * (y+1) + w * h * z;
//     int zi = x + w * y + w * h * (z+1);

//     float x1 = 0.f;
//     float x2 = 0.f;
//     float x3 = 0.f;

//     if (x < w && y < h && z < l)
//     {
//         float val = xbar[i];
//         x1 = (x + 1 < w) ? (xbar[xi] - val) : 0.f;
//         x2 = (y + 1 < h) ? (xbar[yi] - val) : 0.f;
//         x1 = (z + 1 < l) ? (xbar[zi] - val) : 0.f;
//         dx[i] = y1[i] + sigma * x1;
//         dy[i] = y2[i] + sigma * x2;
//         dz[i] = y3[i] + sigma * x3;
//     }

// int i = x + w * y + w * h * z;
//     int xi = (x+1) + w * y + w * h * z;
//     int yi = x + w * (y+1) + w * h * z;
//     int zi = x + w * y + w * h * (z+1);

//     if (x < w && y < h && z < l)
//     {
//         float val = xbar[i];
//         dx[i] = y1[i] + sigma * (xbar[min(max(0, xi), w-1)] - val);
//         dy[i] = y2[i] + sigma * (xbar[min(max(0, yi), h-1)] - val);
//         dz[i] = y3[i] + sigma * (xbar[min(max(0, zi), l-1)] - val);
//     }

__global__ void gradient(float* dx, float* dy, float* dz, float* y1, float* y2, float* y3, float* xbar, float sigma, int w, int h, int l)
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
        dx[x + w * y + w * h * z] = y1[x + w * y + w * h * z] + sigma * x1;
        dy[x + w * y + w * h * z] = y2[x + w * y + w * h * z] + sigma * x2;
        dz[x + w * y + w * h * z] = y3[x + w * y + w * h * z] + sigma * x3;
    }
}

__global__ void clipping(float* xn, float* xcur, float* y1, float* y2, float* y3, float tau, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        // float d1 = ((x>0 ? y1[(x-1) + w * y + w * h * z] : 0.f) - (x+1<w ? y1[x + w * y + w * h * z] : 0.f));
        // float d2 = ((y>0 ? y2[x + w * (y-1) + w * h * z] : 0.f) - (y+1<h ? y2[x + w * y + w * h * z] : 0.f));
        // float d3 = ((z>0 ? y3[x + w * y + w * h * (z-1)] : 0.f) - (z+1<l ? y3[x + w * y + w * h * z] : 0.f));
        float d1 = y1[x + w * y + w * h * z] - (x>0 ? y1[(x-1) + w * y + w * h * z] : 0.f);
        float d2 = y2[x + w * y + w * h * z] - (y>0 ? y2[x + w * (y-1) + w * h * z] : 0.f);
        float d3 = y3[x + w * y + w * h * z] - (z>0 ? y3[x + w * y + w * h * (z-1)] : 0.f);
        // d1 = y1[i] - y1[min(max(0, xi), w-1)];
        // d2 = y2[i] - y2[min(max(0, yi), h-1)];
        // d3 = y3[i] - y3[min(max(0, zi), l-1)];
        float val = xcur[x + w * y + w * h * z] + tau * (d1 + d2 + d3);
        // xn[x + w * y + w * h * z] = val;
        if (z == 0) {
            xn[x + w * y + w * h * z] = 1.f;
        } else if (z == l-1) {
            xn[x + w * y + w * h * z] = 0.f;
        } else {
            xn[x + w * y + w * h * z] = fmin(1.f, fmax(0.f, val));
        }
    }
}

__global__ void extrapolate(float* xbar, float* xcur, float* xn, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x < w && y < h && z < l) {
        xbar[x + w * y + w * h * z] = 2.f * xn[x + w * y + w * h * z] - xcur[x + w * y + w * h * z];
        xcur[x + w * y + w * h * z] = xn[x + w * y + w * h * z];
        // float val = xn[i];
        // xbar[i] = 2.f * val - xcur[i];
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
    int level = 16;
    getParam("level", level, argc, argv);
    cout << "level: " << level << endl;

    // load the input image as grayscale if "-gray" is specifed
    float L = sqrtf(12);
    getParam("L", L, argc, argv);
    cout << "L: " << L << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    float sigma = 1.f/L;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma: " << sigma << endl;

    // load the input image as grayscale if "-gray" is specifed
    float tau = 1.f/L;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;

    // load the input image as grayscale if "-gray" is specifed
    float lambda = 0.1;
    getParam("lambda", lambda, argc, argv);
    cout << "lambda: " << lambda << endl;

    // load the input image as grayscale if "-gray" is specifed
    float nu = 5.f;
    getParam("nu", nu, argc, argv);
    cout << "nu: " << nu << endl;

    // load the input image as grayscale if "-gray" is specifed
    float delta = 1e-9;
    getParam("delta", delta, argc, argv);
    cout << "delta: " << delta << endl;

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
    int dim = w*h*nc;
    int size = w*h*nc*level;
    int projections = level * (level+1) / 2 + 1;
    int nbytes = size*sizeof(float);
    int nbyted = dim*sizeof(float);
    int nbytep = projections*size*sizeof(float);
    cout << "image: " << w << " x " << h << endl;

    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    // allocate raw input image array
    float* h_imgIn  = new float[(size_t)dim];
    float* h_imgOut = new float[(size_t)dim];
    float* h_x1 = new float[(size_t)size];
    float* h_x2 = new float[(size_t)size];
    float* h_x3 = new float[(size_t)size];
    float* h_norm = new float[(size_t)size];

    // allocate raw input image for GPU
    float* d_imgIn; cudaMalloc(&d_imgIn, nbyted); CUDA_CHECK;
    float* d_imgOut;cudaMalloc(&d_imgOut, nbyted); CUDA_CHECK;

    float* d_x; cudaMalloc(&d_x, nbytes); CUDA_CHECK;
    float* d_xbar; cudaMalloc(&d_xbar, nbytes); CUDA_CHECK;
    float* d_xcur; cudaMalloc(&d_xcur, nbytes); CUDA_CHECK;

    float* d_delX; cudaMalloc(&d_delX, nbytes); CUDA_CHECK;
    float* d_delY; cudaMalloc(&d_delY, nbytes); CUDA_CHECK;
    float* d_delZ; cudaMalloc(&d_delZ, nbytes); CUDA_CHECK;

    float* d_y1; cudaMalloc(&d_y1, nbytes); CUDA_CHECK;
    float* d_y2; cudaMalloc(&d_y2, nbytes); CUDA_CHECK;
    float* d_y3; cudaMalloc(&d_y3, nbytes); CUDA_CHECK;

    float* d_u1; cudaMalloc(&d_u1, nbytes); CUDA_CHECK;
    float* d_u2; cudaMalloc(&d_u2, nbytes); CUDA_CHECK;
    float* d_u3; cudaMalloc(&d_u3, nbytes); CUDA_CHECK;

    float* d_tmp1; cudaMalloc(&d_tmp1, nbytes); CUDA_CHECK;
    float* d_tmp2; cudaMalloc(&d_tmp2, nbytes); CUDA_CHECK;
    float* d_tmp3; cudaMalloc(&d_tmp3, nbytes); CUDA_CHECK;

    float* d_r1; cudaMalloc(&d_r1, nbytes); CUDA_CHECK;
    float* d_r2; cudaMalloc(&d_r2, nbytes); CUDA_CHECK;
    float* d_r3; cudaMalloc(&d_r3, nbytes); CUDA_CHECK;

    float* d_v1; cudaMalloc(&d_v1, nbytep); CUDA_CHECK;
    float* d_v2; cudaMalloc(&d_v2, nbytep); CUDA_CHECK;
    float* d_v3; cudaMalloc(&d_v3, nbytep); CUDA_CHECK;

    float* d_norm; cudaMalloc(&d_norm, nbytes); CUDA_CHECK;

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
    cudaMemcpy(d_imgIn, h_imgIn, nbyted, cudaMemcpyHostToDevice); CUDA_CHECK;

    // launch kernel
    // dim3 block = dim3(32, 8, nc);
    // dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1) / block.z);
    dim3 block = dim3(32, 8, 4);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (level + block.z - 1) / block.z);
    dim3 block_iso = dim3(32, 8, 1);
    dim3 grid_iso = dim3((w + block_iso.x - 1) / block_iso.x, (h + block_iso.y - 1) / block_iso.y, 1);

    Timer timer; timer.start();

    int count_p = projections;
    // float sum = 0.f;

    init <<<grid_iso, block_iso>>> (d_xbar, d_xcur, d_x, d_y1, d_y2, d_y3, d_imgIn, w, h, level);
    // cudaMemcpy(h_x1, d_xcur, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
    // Print1D(h_x1, w, h, level);
    
    for (int i = 0; i < repeats; i++)
    {
        cout << "Step " << i+1 << " of " << repeats << " with count_p = " << count_p << endl;
        
        gradient <<<grid, block>>> (d_delX, d_delY, d_delZ, d_y1, d_y2, d_y3, d_xbar, sigma, w, h, level);
        // cudaMemcpy(h_x1, d_delY, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // cudaMemcpy(h_x2, d_delX, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // cudaMemcpy(h_x3, d_delZ, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // Print3D(h_x1, h_x2, h_x3, w, h, level);

        set_u_v <<<grid, block>>> (d_u1, d_u2, d_u3, d_v1, d_v2, d_v3, d_delX, d_delY, d_delZ, w, h, level, projections);
        
        for (int j = 0; j < dykstra; j++)
        {
            count_p = 0;

            // set_r <<<grid, block>>> (d_r1, d_r2, d_r3, d_u1, d_u2, d_u3, w, h, level);
            
            set_tmp_u <<<grid, block>>> (d_tmp1, d_tmp2, d_tmp3, d_u1, d_u2, d_u3, d_v1, d_v2, d_v3, w, h, level, count_p);
            project_on_parabola <<<grid, block>>> (d_u1, d_u2, d_u3, d_tmp1, d_tmp2, d_tmp3, d_imgIn, L, lambda, w, h, level);
            update_v <<<grid, block>>> (d_v1, d_v2, d_v3, d_u1, d_u2, d_u3, d_tmp1, d_tmp2, d_tmp3, w, h, level, count_p);
            count_p++;
            
            for (int k1 = 0; k1 < level; k1++)
            {
                for (int k2 = k1; k2 < level; k2++)
                {
                    set_tmp_u <<<grid, block>>> (d_tmp1, d_tmp2, d_tmp3, d_u1, d_u2, d_u3, d_v1, d_v2, d_v3, w, h, level, count_p);
                    soft_shrinkage <<<grid_iso, block_iso>>> (d_u1, d_u2, d_u3, d_tmp1, d_tmp2, d_tmp3, nu, k1, k2, w, h, level);
                    update_v <<<grid, block>>> (d_v1, d_v2, d_v3, d_u1, d_u2, d_u3, d_tmp1, d_tmp2, d_tmp3, w, h, level, count_p);
                    count_p++;
                }
            }

            // calc_norm <<<grid, block>>> (d_norm, d_r1, d_r2, d_r3, d_u1, d_u2, d_u3, w, h, level);

            // cudaMemcpy(h_norm, d_norm, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;

            // sum = 0.f;
            // for (int v = 0; v < size; v++) {
            //     sum+=h_norm[v]*h_norm[v];
            // }
            // // cout << "Current estimate: " << sum << endl;
            // if (sum < delta) {
            //     break;
            // }
        }
        
        set_y <<<grid, block>>> (d_y1, d_y2, d_y3, d_u1, d_u2, d_u3, w, h, level, projections);
        
        clipping <<<grid, block>>> (d_x, d_xcur, d_y1, d_y2, d_y3, tau, w, h, level);
        // clipping <<<grid, block>>> (d_x, d_xcur, d_delX, d_delY, d_delZ, tau, w, h, level);
        // cudaMemcpy(h_x1, d_xbar, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // Print1D(h_x1, w, h, level);
        
        extrapolate <<<grid, block>>> (d_xbar, d_xcur, d_x, w, h, level);
    }
    // cudaMemcpy(h_x1, d_x, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
    // Print1D(h_x1, w, h, level);

    isosurface <<<grid_iso, block_iso>>> (d_imgOut, d_x, w, h, level);
    // cudaMemcpy(h_imgOut, d_imgOut, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;
    // Print1D(h_imgOut, w, h, 1);

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    cudaMemcpy(h_imgOut, d_imgOut, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;

    // free GPU m.femory
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    
    cudaFree(d_x); CUDA_CHECK;
    cudaFree(d_xbar); CUDA_CHECK;
    cudaFree(d_xcur); CUDA_CHECK;

    cudaFree(d_delX); CUDA_CHECK;
    cudaFree(d_delY); CUDA_CHECK;
    cudaFree(d_delZ); CUDA_CHECK;

    cudaFree(d_y1); CUDA_CHECK;
    cudaFree(d_y2); CUDA_CHECK;
    cudaFree(d_y3); CUDA_CHECK;

    cudaFree(d_u1); CUDA_CHECK;
    cudaFree(d_u2); CUDA_CHECK;
    cudaFree(d_u3); CUDA_CHECK;

    cudaFree(d_tmp1); CUDA_CHECK;
    cudaFree(d_tmp2); CUDA_CHECK;
    cudaFree(d_tmp3); CUDA_CHECK;

    cudaFree(d_r1); CUDA_CHECK;
    cudaFree(d_r2); CUDA_CHECK;
    cudaFree(d_r3); CUDA_CHECK;

    cudaFree(d_v1); CUDA_CHECK;
    cudaFree(d_v2); CUDA_CHECK;
    cudaFree(d_v3); CUDA_CHECK;

    cudaFree(d_norm); CUDA_CHECK;

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
    delete[] h_x1;
    delete[] h_x2;
    delete[] h_x3;
    delete[] h_norm;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}