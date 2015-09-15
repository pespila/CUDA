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
using namespace std;

// uncomment to use the camera
// #define CAMERA

__device__ float interpolate(float k, float uk0, float uk1, float l)
{
    return (k + (0.5 - uk0) / (uk1 - uk0)) / l;
}

inline __device__ float l2Norm(float x1, float x2)
{
    return sqrtf(x1*x1 + x2*x2);
}

inline __device__ float bound(float x1, float x2, float lambda, float k, float L, float f)
{
    return 0.25f * (x1*x1 + x2*x2) - lambda * pow(k / L - f, 2);
}

__device__ void project(float* x1, float* x2, float* x3, const float* img, float L, float lambda, float k, int i, int il)
{
    float f = img[i];
    float y = x3[il] + lambda * pow(k / L - f, 2);
    float norm = l2Norm(x1[il], x2[il]);
    float v = 0.f;
    float a = 2.f * 0.25f * norm;
    float b = 2.f / 3.f * (1.f - 2.f * 0.25f * y);
    float d = b < 0 ? (a - pow(sqrt(-b), 3)) * (a + pow(sqrt(-b), 3)) : a * a + b * b * b;
    float c = pow((a + sqrt(d)), 1.f / 3.f);
    if (d >= 0) {
        v = c != 0 ? c - b / c : 0.f;
    } else {
        v = 2.f * sqrt(-b) * cos((1.f / 3.f) * acos(a / (pow(sqrt(-b), 3))));
    }
    x1[il] = norm != 0 ? (v / (2.0 * 0.25f)) * x1[il] / norm : 0.f;
    x2[il] = norm != 0 ? (v / (2.0 * 0.25f)) * x2[il] / norm : 0.f;
    x3[il] = bound(x1[il], x2[il], lambda, k, L, img[i]);
}

__global__ void parabolaProjection(float* x1, float* x2, float* x3, const float* img, float L, float lambda, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int i = x + w * y;
    if (x < w && y < h)
    {
        int il;
        float bound_val;
        for (int k = 0; k < l; k++)
        {
            il = i + k * w * h;
            bound_val = bound(x1[il], x2[il], lambda, k, L, img[i]);
            if (x3[il] < bound_val)
            {
                project(x1, x2, x3, img, L, lambda, k, i, il);
            }
        }
    }
}
//     void SoftShrinkage(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, int i, int j, int k, int k1, int k2, F nu) {
//         if (dst.Size() != 3) {
//             cout << "ERROR 10 (SoftShrinkage): Size of dst does not match!" << endl;
//         } else {
//             F K = (F)(k2 - k1 + 1);
//             primaldual::Vector<F> s(1, 2, 0.0);
//             primaldual::Vector<F> s0(1, 2, 0.0);
//             for (int l = k1; l <= k2; l++) {
//                 for (int c = 0; c < 2; c++) {
//                     s0.Set(0, c, s0.Get(0, c) + src.Get(i, j, l, c));
//                 }
//             }
//             L2(s, s0, nu);
//             for (int c = 0; c < 2; c++) {
//                 if (k >= k1 && k <= k2) {
//                     dst.Set(0, c, src.Get(i, j, k, c) + ((s.Get(0, c) - s0.Get(0, c)) / K));
//                 } else {
//                     dst.Set(0, c, src.Get(i, j, k, c));
//                 }
//             }
//         }
//     }

// __global__ softShrinkage(float* x1, float* x2, float* x3, float nu, int k1, int k2, int w, int h, int l)
// {
//     int x = threadIdx.x + blockDim.x * blockIdx.x;
//     int y = threadIdx.y + blockDim.y * blockIdx.y;

//     int i = x + w * y;
//     float K = (float)(k2 - k1 + 1);

//     if (x < w && y < h)
//     {
//         float s1 = 0.f;
//         float s2 = 0.f;
//         float s01 = 0.f;
//         float s02 = 0.f;

//         for (int k = k1; k <= k2; k++)
//         {
//             s01 += x1[i + j * w * h];
//             s02 += x2[i + j * w * h];
//         }

//         float norm = l2Norm(s01, s02);
//         if (norm <= nu)
//         {
//             s1 = s01;
//             s2 = s02;
//         } else {
//             s1 = s01 / norm;
//             s2 = s02 / norm;
//         }

//         if ()
//     }
// }

__global__ void init(float* xbar, float* xcur, float* xn, float* img, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int i = x + w * y;
    if (x < w && y < h)
    {
        float img_val = img[i];
        float t = 0.f;
        float val = 0.f;
        for (int k = 0; k < l; k++)
        {
            t = (float)k / (float)l;
            if (t <= img_val)
                val = 1.f;
            else
                val = 0.f;
            xn[i + k * w * h] = val;
            xcur[i + k * w * h] = val;
            xbar[i + k * w * h] = val;
        }
    }
}

__global__ void isosurface(float* img, float* xbar, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int i = x + w * y;
    int k = 0;
    if (x < w && y < h)
    {
        float val = 0.f;
        float uk0 = 0.f;
        float uk1 = 0.f;
        
        while (k < l-1)
        {
            uk0 = xbar[i + k * w * h];
            uk1 = xbar[i + (k+1) * w * h];
            if (uk0 > 0.5 && uk1 <= 0.5)
            {
                val = interpolate(k, uk0, uk1, l);
                k = l;
            } else {
                k++;
            }
        }

        if (k == l)
            img[i] = val;
        else
            img[i] = uk1;
    }
}

__global__ void gradient(float* dx, float* dy, float* dz, float* y1, float* y2, float* y3, float* v, float sigma, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    int i = x + w * y + w * h * z;
    int xi = (x+1) + w * y + w * h * z;
    int yi = x + w * (y+1) + w * h * z;
    int zi = x + w * y + w * h * (z+1);

    if (x < w && y < h && z < l)
    {
        float val = v[i];
        dx[i] = y1[i] + sigma * (v[min(max(0, xi), w-1)]-val);
        dy[i] = y2[i] + sigma * (v[min(max(0, yi), h-1)]-val);
        dz[i] = y3[i] + sigma * (v[min(max(0, zi), l-1)]-val);
    }
}

__global__ void clipping(float* xn, float* y1, float* y2, float* y3, float* xcur, float tau, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    int i = x + w * y + w * h * z;
    int xi = (x-1) + w * y + w * h * z;
    int yi = x + w * (y-1) + w * h * z;
    int zi = x + w * y + w * h * (z-1);

    if (x < w && y < h && z < l)
    {
        float d1, d2, d3, val;
        d1 = y1[i]-y1[min(max(0, xi), w-1)];
        d2 = y2[i]-y2[min(max(0, yi), h-1)];
        d3 = y3[i]-y3[min(max(0, zi), l-1)];
        val = xcur[i] + tau * (d1 + d2 + d3);
        xn[i] = fmin(1.f, fmax(0.f, val));
    }
}

__global__ void extrapolate(float* xbar, float* xn, float* xcur, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    int i = x + w * y + w * h * z;

    if (x < w && y < h && z < l) {
        float val = xn[i];
        xbar[i] = 2.f * val - xcur[i];
        xcur[i] = val;
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
    int level = 8;
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
    int nbytes = size*sizeof(float);
    int nbyted = dim*sizeof(float);
    cout << "image: " << w << " x " << h << endl;

    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    // allocate raw input image array
    float* h_imgIn  = new float[(size_t)dim];
    float* h_imgOut = new float[(size_t)dim];

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
    dim3 grid_iso = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    Timer timer; timer.start();

    init <<<grid_iso, block_iso>>> (d_xbar, d_xcur, d_x, d_imgIn, w, h, level);
    for (int i = 0; i < repeats; i++)
    {
        gradient <<<grid, block>>> (d_delX, d_delY, d_delZ, d_y1, d_y2, d_y3, d_xbar, sigma, w, h, level);
        for (int j = 0; j < dykstra; j++)
        {
        //     boyle_dykstra <<<grid, block>>> (d_y1, d_y2, d_y3, d_delX, d_delY, d_delZ, d_imgIn, L, lambda, w, h, level);
        }
        clipping <<<grid, block>>> (d_x, d_y1, d_y2, d_y3, d_xcur, tau, w, h, level);
        extrapolate <<<grid, block>>> (d_xbar, d_x, d_xcur, w, h, level);
    }
    isosurface <<<grid_iso, block_iso>>> (d_imgOut, d_xbar, w, h, level);

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    // cudaMemcpy(h_imgOut, d_imgIn, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(h_imgOut, d_imgOut, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;

    // free GPU memory
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