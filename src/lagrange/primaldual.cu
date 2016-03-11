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

void parameterToFile(string filename,int repeats,bool gray,int level,float tauu,float taum,float sigmap,float sigmas,float lambda,float nu,int w,int h,int nc,size_t available,size_t total,float t,int iter) {
    FILE *file;
    file = fopen(filename.c_str(), "w");
    if(file == NULL)
        printf("ERROR: Could not open file!");
    else {
        fprintf(file, "image: %d x %d x %d\n", w, h, nc);
        fprintf(file,"repeats: %d\n", repeats);
        fprintf(file,"gray: %d\n", gray);
        fprintf(file,"level: %d\n", level);
        fprintf(file,"tauu: %f\n", tauu);
        fprintf(file,"taum: %f\n", taum);
        fprintf(file,"sigmas: %f\n", sigmas);
        fprintf(file,"lambda: %f\n", lambda);
        fprintf(file,"nu: %f\n", nu);
        fprintf(file, "GPU Memory: %zd - %zd = %f GB\n", total, available, (total-available)/pow(10,9));
        fprintf(file, "time: %f s\n", t);
        fprintf(file, "iterations: %d\n", iter);
    }
    fclose (file);
}

void parameterToConsole(string filename,int repeats,bool gray,int level,float tauu,float taum,float sigmap,float sigmas,float lambda,float nu,int w,int h,int nc,size_t available,size_t total,float t,int iter) {
    printf( "image: %d x %d x %d\n", w, h, nc);
    printf("repeats: %d\n", repeats);
    printf("gray: %d\n", gray);
    printf("level: %d\n", level);
    printf("tauu: %f\n", tauu);
    printf("taum: %f\n", taum);
    printf("sigmas: %f\n", sigmas);
    printf("lambda: %f\n", lambda);
    printf("nu: %f\n", nu);
    printf( "GPU Memory: %zd - %zd = %f GB\n", total, available, (total-available)/pow(10,9));
    printf( "time: %f s\n", t);
    printf( "iterations: %d\n", iter);
}

float energy(float* u, float* un, int size) {
    float nrj = 0.f;
    for (int i = 0; i < size; i++)
        nrj += fabs(u[i] - un[i]);
    return nrj;
}

__device__ float bound(float x1, float x2, float lambda, float k, float l, float f)
{
    return 0.25f * (x1*x1 + x2*x2) - lambda * pow(k / l - f, 2);
}

__device__ float interpolate(float k, float uk0, float uk1, float l)
{
    return (k + (0.5 - uk0) / (uk1 - uk0)) / l;
}

__device__ void on_parabola(float* u1,float* u2,float* u3,float x1,float x2,float x3,float f,float lambda,float k,int j,float l)
{
    float y = x3 + lambda * pow(k / l - f, 2);
    float norm = sqrtf(x1*x1+x2*x2);
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

__global__ void init(float* u,float* ubar,float* p1,float* p2,float* p3,float* s1,float* s2,float* mu1,float* mu2,float* mubar1,float* mubar2,float* f,int h,int w,int l,int proj,int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h)
    {
        int I, J;
        float img;

        for (int c = 0; c < nc; c++)
        {
            img = f[x+y*w+c*h*w]; // image value
            for (int k = 0; k < proj; k++)
            {
                I = x+y*w+k*h*w+c*h*w*l; // index for u, ubar, p1, p2, p3
                J = x+y*w+k*h*w+c*h*w*proj; // index for s1, s2, mu1, mu2, mubar1, mubar2
                if (k<l) {
                    u[I] = img;
                    ubar[I] = img;
                    p1[I] = 0.f;
                    p2[I] = 0.f;
                    p3[I] = 0.f;
                }
                s1[J] = 0.f;
                s2[J] = 0.f;
                mu1[J] = 0.f;
                mu2[J] = 0.f;
                mubar1[J] = 0.f;
                mubar2[J] = 0.f;
            }
        }
    }
}

__global__ void parabola(float* p1,float* p2,float* p3,float* mu1,float* mu2,float* ubar,float* f,float sigma,float lambda,int w,int h,int l,int proj,int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        for (int c = 0; c < nc; c++)
        {
            int I = x+y*w+c*h*w; // index for image
            int J = x+y*w+z*h*w+c*h*w*l; // index for

            float B;
            
            float img = f[I];
            float mu1sum = 0.f;
            float mu2sum = 0.f;

            float val = ubar[J];
            float u1 = (y+1<h) ? (ubar[x+(y+1)*w+z*h*w+c*h*w*l]-val) : 0.f;
            float u2 = (x+1<w) ? (ubar[(x+1)+y*w+z*h*w+c*h*w*l]-val) : 0.f;
            float u3 = (z+1<l) ? (ubar[x+y*w+(z+1)*h*w+c*h*w*l]-val) : 0.f;

            int K = 0;
            for (int k1 = 0; k1 < l; k1++)
            {
                for (int k2 = k1; k2 < l; k2++)
                {
                    if (z <= k2 && z >= k1) {
                        mu1sum+=mu1[x+y*w+K*h*w+c*h*w*proj];
                        mu2sum+=mu2[x+y*w+K*h*w+c*h*w*proj];
                    }
                    K++;
                }
            }

            u1 = p1[J]+sigma*(u1+mu1sum);
            u2 = p2[J]+sigma*(u2+mu2sum);
            u3 = p3[J]+sigma*u3;

            B = bound(u1,u2,lambda,z+1,l,img);
            if (u3 < B) {
                on_parabola(p1,p2,p3,u1,u2,u3,img,lambda,z+1,J,l);
            } else {
                p1[J] = u1;
                p2[J] = u2;
                p3[J] = u3;
            }
        }
    }
}

__global__ void l2projection(float* s1,float* s2,float* mubar1,float* mubar2,float sigma,float nu,int w,int h,int l,int proj,int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h)
    {
        float m1, m2;
        float norm;
        int I;
        for (int c = 0; c < nc; c++)
        {
            for (int k = 0; k < proj; k++)
            {
                I = x+y*w+k*h*w+c*h*w*proj;
                m1=s1[I]-sigma*mubar1[I];
                m2=s2[I]-sigma*mubar2[I];
                norm=sqrtf(m1*m1+m2*m2);
                s1[I]=(norm<=nu) ? m1 : nu*m1/norm;
                s2[I]=(norm<=nu) ? m2 : nu*m2/norm;
            }
        }
    }
}

__global__ void clipping(float* u,float* ubar,float* p1,float* p2,float* p3,float tau,int w,int h,int l,int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        int I;
        float cur;
        float d1,d2,d3,D;
        for (int c = 0; c < nc; c++)
        {
            I = x+y*w+z*h*w+c*h*w*l;
            cur = u[I];
            d1 = (y+1<h ? p1[I] : 0.f) - (y>0 ? p1[x+(y-1)*w+z*h*w+c*h*w*l] : 0.f);
            d2 = (x+1<w ? p2[I] : 0.f) - (x>0 ? p2[(x-1)+y*w+z*h*w+c*h*w*l] : 0.f);
            d3 = (z+1<l ? p3[I] : 0.f) - (z>0 ? p3[x+y*w+(z-1)*h*w+c*h*w*l] : 0.f);
            D = cur+tau*(d1+d2+d3);
            if (z==0) {
                u[I]=1.f;
            } else if (z==l-1) {
                u[I]=0.f;
            } else {
                u[I]=fmin(1.f, fmax(0.f, D));
            }
            ubar[I] = 2.f * u[I] - cur;
        }
    }
}

__global__ void mu(float* mu1,float* mu2,float* mubar1,float* mubar2,float* s1,float* s2,float* p1,float* p2,int w,int h,int l,int proj,int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h)
    {
        float tau = 1.f / (2.f + (float)(proj/4.f));
        float t1, t2, c1, c2;
        int I, J, K;
        for (int c = 0; c < nc; c++)
        {
            K = 0;
            for (int k1 = 0; k1 < l; k1++)
            {
                for (int k2 = k1; k2 < l; k2++)
                {
                    I = x+y*w+K*h*w+c*h*w*proj;
                    c1 = mu1[I]; c2 = mu2[I];
                    t1 = 0.f; t2 = 0.f;
                    for (int k = k1; k <= k2; k++)
                    {
                        J = x+y*w+k*h*w+c*h*w*l;
                        t1+=p1[J];
                        t2+=p2[J];
                    }
                    mu1[I] = c1+tau*(s1[I]-t1);
                    mu2[I] = c2+tau*(s2[I]-t2);
                    mubar1[I] = 2.f * mu1[I] - c1;
                    mubar2[I] = 2.f * mu2[I] - c2;
                    K++;
                }
            }
        }
    }
}

__global__ void isosurface(float* f,float* u,int w,int h,int l,int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < w && y < h)
    {
        float uk0, uk1, val;
        for (int c = 0; c < nc; c++)
        {
            for (int k = 0; k < l-1; k++)
            {
                uk0 = u[x+y*w+k*h*w+c*h*w*l];
                uk1 = u[x+y*w+(k+1)*h*w+c*h*w*l];
                if (uk0 > 0.5 && uk1 <= 0.5) {
                    val = interpolate(k+1, uk0, uk1, l);
                    break;
                } else {
                    val = uk1;
                }
            }
            f[x+y*w+c*h*w] = val;
        }
    }
}

int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;

    if (argc <= 3) { cout << "Usage: " << argv[0] << " -i <image> -o <output_image> -parm <parameter.txt> [-repeats <repeats>] [-gray]" << endl; return 1; }
    
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;

    // output image
    string output = "";
    bool retO = getParam("o", output, argc, argv);
    if (!retO) cerr << "ERROR: no output image specified" << endl;

    // parameter values
    string parm = "";
    bool ret2 = getParam("parm", parm, argc, argv);

    // number of computation repetitions to get a better run time measurement
    int repeats = 1000;
    getParam("repeats", repeats, argc, argv);
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    int level = 16;
    getParam("level", level, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    float lambda = 0.1f;
    getParam("lambda", lambda, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    float nu = 0.01f;
    getParam("nu", nu, argc, argv);
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // time-steps
    float nrj;
    float tauu = 1.f / 6.f;
    float sigmap = 1.f / (3.f + level);
    float sigmas = 1.f;

    // get image dimensions
    int iter = 1;
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    
    // image dimension
    int dim = h*w*nc;
    int nbyted = dim*sizeof(float);
    
    // u, un, ubar, p1, p2, p3 dimension
    int size = h*w*level*nc;
    int nbytes = size*sizeof(float);
    
    // s1, s2, mu1, mu2, mun1, mun2, mubar1, mubar2 dimension
    int proj = level*(level-1)/2 + level;
    int nbytep = proj*dim*sizeof(float);

    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    // allocate raw input image array
    // float* h_u = new float[size];
    // float* h_un = new float[size];
    float* h_img  = new float[(size_t)dim];
    float* h_u  = new float[(size_t)size];
    float* h_un  = new float[(size_t)size];

    // allocate raw input image for GPU
    float* d_f; cudaMalloc(&d_f, nbyted); CUDA_CHECK;

    float* d_u; cudaMalloc(&d_u, nbytes); CUDA_CHECK;
    float* d_ubar; cudaMalloc(&d_ubar, nbytes); CUDA_CHECK;

    float* d_p1; cudaMalloc(&d_p1, nbytes); CUDA_CHECK;
    float* d_p2; cudaMalloc(&d_p2, nbytes); CUDA_CHECK;
    float* d_p3; cudaMalloc(&d_p3, nbytes); CUDA_CHECK;

    float* d_s1; cudaMalloc(&d_s1, nbytep); CUDA_CHECK;
    float* d_s2; cudaMalloc(&d_s2, nbytep); CUDA_CHECK;

    float* d_mu1; cudaMalloc(&d_mu1, nbytep); CUDA_CHECK;
    float* d_mu2; cudaMalloc(&d_mu2, nbytep); CUDA_CHECK;
    float* d_mubar1; cudaMalloc(&d_mubar1, nbytep); CUDA_CHECK;
    float* d_mubar2; cudaMalloc(&d_mubar2, nbytep); CUDA_CHECK;

    size_t available, total;
    cudaMemGetInfo(&available, &total);

    // Init raw input image array
    convert_mat_to_layered (h_img, mIn);
    // copy host memory
    cudaMemcpy(d_f,h_img,nbyted,cudaMemcpyHostToDevice); CUDA_CHECK;

    // launch kernel
    dim3 block = dim3(32, 8, 4);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (level + block.z - 1) / block.z);
    dim3 block_iso = dim3(32, 8, 1);
    dim3 grid_iso = dim3((w + block_iso.x - 1) / block_iso.x, (h + block_iso.y - 1) / block_iso.y, 1);

    Timer timer; timer.start();
    
    init <<<grid_iso, block_iso>>> (d_u,d_ubar,d_p1,d_p2,d_p3,d_s1,d_s2,d_mu1,d_mu2,d_mubar1,d_mubar2,d_f,h,w,level,proj,nc);
    for (iter = 0; iter < repeats; iter++)
    {
        parabola <<<grid, block>>> (d_p1,d_p2,d_p3,d_mu1,d_mu2,d_ubar,d_f,sigmap,lambda,w,h,level,proj,nc);
        l2projection <<<grid_iso, block_iso>>> (d_s1,d_s2,d_mubar1,d_mubar2,sigmas,nu,w,h,level,proj,nc);
        mu <<<grid_iso, block_iso>>> (d_mu1,d_mu2,d_mubar1,d_mubar2,d_s1,d_s2,d_p1,d_p2,w,h,level,proj,nc);
        if (iter%10 == 0) cudaMemcpy(h_un, d_u, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        clipping <<<grid, block>>> (d_u,d_ubar,d_p1,d_p2,d_p3,tauu,w,h,level,nc);
        if (iter%10 == 0) {
            cudaMemcpy(h_u, d_u, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
            nrj = energy(h_u, h_un, size);
            if (nrj/(w*h*level) <= 5*1E-5) break;
        }
    }
    isosurface <<<grid_iso, block_iso>>> (d_f,d_u,w,h,level,nc);

    timer.end();  float t = timer.get();  // elapsed time in seconds

    cudaMemcpy(h_img,d_f,nbyted,cudaMemcpyDeviceToHost); CUDA_CHECK;

    if (!ret2) {
        parameterToConsole(parm,repeats,gray,level,tauu,1.f,sigmap,sigmas,lambda,nu,w,h,nc,available,total,t,iter);
    } else {
        parameterToFile(parm,repeats,gray,level,tauu,1.f,sigmap,sigmas,lambda,nu,w,h,nc,available,total,t,iter);
    }

    // show output image: first convert to interleaved opencv format from the layered raw array
    // showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
    convert_layered_to_mat(mOut,h_img);
    // showImage("Output", mOut, 100+w+40, 100);

    // save input and result
    cv::imwrite(output,mOut*255.f);

    // free GPU memory
    cudaFree(d_f); CUDA_CHECK;
    
    cudaFree(d_u); CUDA_CHECK;
    cudaFree(d_ubar); CUDA_CHECK;

    cudaFree(d_p1); CUDA_CHECK;
    cudaFree(d_p2); CUDA_CHECK;
    cudaFree(d_p3); CUDA_CHECK;

    cudaFree(d_s1); CUDA_CHECK;
    cudaFree(d_s2); CUDA_CHECK;

    cudaFree(d_mu1); CUDA_CHECK;
    cudaFree(d_mu2); CUDA_CHECK;
    cudaFree(d_mubar1); CUDA_CHECK;
    cudaFree(d_mubar2); CUDA_CHECK;

    // free allocated arrays
    delete[] h_u;
    delete[] h_un;
    delete[] h_img;
    return 0;
}