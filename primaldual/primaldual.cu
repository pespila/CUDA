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
                cout << "(";
                // for (int c = 0; c < Dimension(); c++) {
                    // printf("%g ", Get(i, j, k, c));
                    cout << x[j + i * w + k * h * w] << "";
                    // cout << (int)(x[j + i * w + k * h * w]*255) << "";
                // }
                cout << ")" << "";
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
                // printf("(%2.g,%2.g,%2.g)", x1[j + i * w + k * h * w], x2[j + i * w + k * h * w], x3[j + i * w + k * h * w]);
                cout << "(";
                cout << x1[j + i * w + k * h * w] << " " << x2[j + i * w + k * h * w] << " " << x3[j + i * w + k * h * w] << "";
                cout << ")" << "";
            }
            // printf("\n");
            cout << endl;
        }
        cout << "_____          _____" << endl << endl;
    }
}

void Print4D(float* x1, float* x2, float* x3, int w, int h, int l, int p) {
    for (int k = 0; k < 1; k++) {
        cout << "_____ Level: " << k << " _____" << endl;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cout << "(";
                cout << x1[j + i * w + k * h * w + p * w * h * l] << " " << x2[j + i * w + k * h * w + p * w * h * l] << " " << x3[j + i * w + k * h * w + p * w * h * l] << "";
                cout << ")" << "";
                // printf("(%2.g,%2.g,%2.g)", x1[j + i * w + k * h * w], x2[j + i * w + k * h * w], x3[j + i * w + k * h * w]);
            }
            // printf("\n");
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
    return (k + (0.5 - uk0) / (uk1 - uk0)) / (l-1);
}

__device__ void on_parabola(float* u1, float* u2, float* u3, float x1, float x2, float x3, float f, float L, float lambda, float k, int j)
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
    u1[j] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x1 / norm;
    u2[j] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x2 / norm;
    u3[j] = bound(u1[j], u2[j], lambda, k, L, f);
}

// /**
// * @brief Computes the orthogonal projection of (x0, y0) onto the epigraph of
// *        the parabola y >= \alpha x^2.
// */
// template<typename T>
// inline __device__ void ProjectParabolaSimple(const T& x0,
//                                             const T& y0,
//                                             const T& alpha,
//                                             T& x,
//                                             T& y)
// {
//  // nothing to do?
//  if(y0 >= alpha * (x0 * x0)) {
//    x = x0;
//    y = y0;
//  }
//  else {
//    const T a = 2. * alpha * fabs(x0);
//    const T b = 2. * (1. - 2. * alpha * y0) / 3.;
//    T d, v;
   
//    if(b < 0) {
//      const T sq = powf(-b, 3. / 2.);
//      d = (a - sq) * (a + sq);      
//    }
//    else {
//      d = a * a + b * b * b;
//    }
   
//    if(d >= 0) {
//      const T c = powf(a + sqrtf(d), 1. / 3.);
//      v = c - b / c;
//    }
//    else {
//      v = 2 * sqrtf(-b) * cos(acos(a / powf(-b, 3. / 2.)) / 3.);
//    }

//    if(x0 > 0)
//      x = v / (2. * alpha);
//    else if(x0 < 0)
//      x = -v / (2. * alpha);
//    else
//      x = 0;

//    y = alpha * x * x;
//  }
// }

// /**
// * @brief Computes orthogonal projection of (x0, y0) onto the epigraph of the
// *        parabola y >= p * x^2 + q * x + r.
// */
// template<typename T>
// inline __device__ void ProjectParabolaGeneral(const T& x0,
//                                              const T& y0,
//                                              const T& p,
//                                              const T& q,
//                                              const T& r,
//                                              T& x,
//                                              T& y)
// {
//  T tildex;
//  T tildey;
 
//  ProjectParabolaSimple<T>(
//      x0 + q / (2. * p),
//      y0 + q * q / (4. * p) - r,
//      p,
//      tildex,
//      tildey);
 
//  x = tildex - q / (2. * p);
//  y = tildey - q * q / (4. * p) + r;
// }
__global__ void project_on_parabola(float* u1, float* u2, float* u3, float* v1, float* v2, float* v3, float* img, float L, float lambda, int k, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    

    if (x < w && y < h && z < l)
    {
        int index = x + w * y;
        int i = x + w * y + w * h * z + (k-1) * w * h * l;
        int j = x + w * y + w * h * z + k * w * h * l;

        float f = img[index];
        float x1 = u1[i] - v1[j];
        float x2 = u2[i] - v2[j];
        float x3 = u3[i] - v3[j];
        float bound_val = bound(x1, x2, lambda, z+1.f, L, f);

        if (x3 < bound_val) {
            on_parabola(u1, u2, u3, x1, x2, x3, f, L, lambda, z+1.f, j);
            // ProjectParabolaGeneral(0.288675f, 0.f, 0.25f, 0.f, bound_val, x1, x3);
        } else {
            u1[j] = x1;
            u2[j] = x2;
            u3[j] = x3;
        }
    }
}

__device__ float partial(float x)
{
    return 0.5 * x;
}

__global__ void soft_shrinkage(float* u1, float* u2, float* u3, float* v1, float* v2, float* v3, float nu, int k1, int k2, int P, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    const float K = (float)(k2 - k1 + 1);

    if (x < w && y < h)
    {
        int i, j;
        float s1 = 0.f;
        float s2 = 0.f;
        float s01 = 0.f;
        float s02 = 0.f;
        float x1 = 0.f;
        float x2 = 0.f;

        for (int k = k1; k <= k2; k++)
        {
            i = x + w * y + k * w * h + (P-1) * w * h * l;
            j = x + w * y + k * w * h + P * w * h * l;
            x1 = u1[i] - v1[j];
            x2 = u2[i] - v2[j];
            s01 += x1;
            s02 += x2;
        }

        float norm = l2Norm(s01, s02);

        s1 = norm <= nu ? s01 : (nu * s01 / norm);
        s2 = norm <= nu ? s02 : (nu * s02 / norm);

        for (int k = 0; k < l; k++)
        {
            i = x + w * y + k * w * h + (P-1) * w * h * l;
            j = x + w * y + k * w * h + P * w * h * l;
            x1 = u1[i] - v1[j];
            x2 = u2[i] - v2[j];
            if (k >= k1 && k <= k2) {
                u1[j] = x1 + (s1 - s01) / K;
                u2[j] = x2 + (s2 - s02) / K;
            } else {
                u1[j] = x1;
                u2[j] = x2;
            }
            u3[j] = u3[i] - v3[j];
        }
    }
}

__global__ void init(float* xbar, float* xcur, float* xn, float* y1, float* y2, float* y3, float* img, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h)
    {
        float img_val = img[x + w * y];
        for (int k = 0; k < l; k++)
        {
            int index = x + w * y + k * w * h;
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

// __global__ void isosurface(float* img, float* xbar, int w, int h, int l)
// {
//     int x = threadIdx.x + blockDim.x * blockIdx.x;
//     int y = threadIdx.y + blockDim.y * blockIdx.y;
    
//     int i = x + w * y;
//     int k = 0;
//     if (x < w && y < h)
//     {
//         float val = 0.f;
//         float uk0 = 0.f;
//         float uk1 = 0.f;
        
//         while (k < l-1)
//         {
//             uk0 = xbar[i + k * w * h];
//             uk1 = xbar[i + (k+1) * w * h];
//             if (uk0 > 0.5 && uk1 <= 0.5)
//             {
//                 val = interpolate(k, uk0, uk1, l);
//                 k = l;
//                 break;
//             } else {
//                 k++;
//             }
//         }

//         if (k == l)
//             img[i] = val;
//         else
//             img[i] = 1.f;
//     }
// }

__global__ void set_y(float* y1, float* y2, float* y3, float* u1, float* u2, float* u3, int w, int h, int l, int p)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    

    if (x < w && y < h && z < l)
    {
        int i = x + w * y + w * h * z;
        int j = x + w * y + w * h * z + (p-1) * w * h * l;
        y1[i] = u1[j];
        y2[i] = u2[j];
        y3[i] = u3[j];
    }
}

__global__ void set_u_v(float* u1, float* u2, float* u3, float* v1, float* v2, float* v3, float* dx, float* dy, float* dz, int w, int h, int l, int p)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        int i = x + w * y + w * h * z;
        int j;
        for (int k = 0; k < p; k++)
        {
            j = x + w * y + w * h * z + k * w * h * l;

            u1[j] = k < p-1 ? 0.f : dx[i];
            u2[j] = k < p-1 ? 0.f : dy[i];
            u3[j] = k < p-1 ? 0.f : dz[i];

            // u1[j] = dx[i];
            // u2[j] = dy[i];
            // u3[j] = dz[i];

            v1[j] = 0.f;
            v2[j] = 0.f;
            v3[j] = 0.f;
            
        }
    }
}

__global__ void update_v(float* v1, float* v2, float* v3, float* u1, float* u2, float* u3, int w, int h, int l, int k)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    

    if (x < w && y < h && z < l)
    {
        int i = x + w * y + w * h * z + k * w * h * l;
        int j = x + w * y + w * h * z + (k-1) * w * h * l;
        v1[i] = u1[i] - (u1[j] - v1[i]);
        v2[i] = u2[i] - (u2[j] - v2[i]);
        v3[i] = u3[i] - (u3[j] - v3[i]);
    }
}

__global__ void set_u_zero(float* u1, float* u2, float* u3, int w, int h, int l, int p)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    

    if (x < w && y < h && z < l)
    {
        int i = x + w * y + w * h * z;
        int j = x + w * y + w * h * z + (p-1) * w * h * l;
        u1[i] = u1[j];
        u2[i] = u2[j];
        u3[i] = u3[j];
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

__global__ void calc_norm(float* r1, float* r2, float* r3, float* u1, float* u2, float* u3, int w, int h, int l, int p)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        int i = x + w * y + w * h * z;
        r1[i] = u1[i] - u1[i + (p-1) * w * h * l];
        r2[i] = u2[i] - u2[i + (p-1) * w * h * l];
        r3[i] = u3[i] - u3[i + (p-1) * w * h * l];    
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

// __global__ void gradient(float* dx, float* dy, float* dz, float* y1, float* y2, float* y3, float* xbar, float sigma, int w, int h, int l)
// {
//     int x = threadIdx.x + blockDim.x * blockIdx.x;
//     int y = threadIdx.y + blockDim.y * blockIdx.y;
//     int z = threadIdx.z + blockDim.z * blockIdx.z;
    
//     int i = x + w * y + w * h * z;
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
// }

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

// __global__ void clipping(float* xn, float* y1, float* y2, float* y3, float tau, int w, int h, int l)
// {
//     int x = threadIdx.x + blockDim.x * blockIdx.x;
//     int y = threadIdx.y + blockDim.y * blockIdx.y;
//     int z = threadIdx.z + blockDim.z * blockIdx.z;
    
//     int i = x + w * y + w * h * z;
//     int xi = (x-1) + w * y + w * h * z;
//     int yi = x + w * (y-1) + w * h * z;
//     int zi = x + w * y + w * h * (z-1);
    
//     float d1, d2, d3, val;

//     if (x < w && y < h && z < l)
//     {
//         d1 = y1[i] - y1[min(max(0, xi), w-1)];
//         d2 = y2[i] - y2[min(max(0, yi), h-1)];
//         d3 = y3[i] - y3[min(max(0, zi), l-1)];
//         val = xn[i] + tau * (d1 + d2 + d3);
        
//         if (z == 0) {
//             xn[i] = 1.f;
//         } else if (z == l-1) {
//             xn[i] = 0.f;
//         } else {
//             xn[i] = fmin(1.f, fmax(0.f, val));
//         }
//     }
// }

__global__ void extrapolate(float* xbar, float* xcur, float* xn, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    int i = x + w * y + w * h * z;

    if (x < w && y < h && z < l) {
        xbar[i] = 2 * xn[i] - xcur[i];
        xcur[i] = xn[i];
        // float val = xn[i];
        // xbar[i] = 2.f * val - xcur[i];
    }
}

int main(int argc, char **argv)
{
    // double TAU = 1.0 / sqrt(12.0);
    // double a = TAU / 2.0;
    // double b = 2.0 / 3.0 - 2.0 / (60.0 * 12.0);
    // double d = a*a + b*b*b;
    // double c = pow((a + sqrt(d)), 1.f/3.f);
    // double v = c - b/c;
    // cout << "x1 = " << 2.0 * v << " x2 = " << 0 << " x3 = " << (2.0*v)*(2.0*v)/4.0 - 0.1*(1.0/sqrt(12.0))*(1.0/sqrt(12.0)) << endl;
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
    // float tau = 1000;
    float tau = 1.f/L;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    float sigma = 1.f/(L*L*tau);
    getParam("sigma", sigma, argc, argv);
    cout << "sigma: " << sigma << endl;

    // load the input image as grayscale if "-gray" is specifed
    float lambda = 0.1;
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
    int nc = 1;  // number of channels
    // int nc = mIn.channels();  // number of channels
    int dim = w*h*nc;
    int size = w*h*nc*level;
    int projections = level * (level+1) / 2 + 1 + 1;
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
    float* h_r = new float[(size_t)size];
    float* h_u1 = new float[(size_t)projections*size];
    float* h_u2 = new float[(size_t)projections*size];
    float* h_u3 = new float[(size_t)projections*size];

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

    float* d_u1; cudaMalloc(&d_u1, nbytep); CUDA_CHECK;
    float* d_u2; cudaMalloc(&d_u2, nbytep); CUDA_CHECK;
    float* d_u3; cudaMalloc(&d_u3, nbytep); CUDA_CHECK;

    float* d_v1; cudaMalloc(&d_v1, nbytep); CUDA_CHECK;
    float* d_v2; cudaMalloc(&d_v2, nbytep); CUDA_CHECK;
    float* d_v3; cudaMalloc(&d_v3, nbytep); CUDA_CHECK;

    // float* d_r1; cudaMalloc(&d_r1, nbytep); CUDA_CHECK;
    // float* d_r2; cudaMalloc(&d_r2, nbytep); CUDA_CHECK;
    // float* d_r3; cudaMalloc(&d_r3, nbytep); CUDA_CHECK;

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
    float sum = 0.f;

    init <<<grid_iso, block_iso>>> (d_xbar, d_xcur, d_x, d_y1, d_y2, d_y3, d_imgIn, w, h, level);
    // cudaMemcpy(h_x1, d_xcur, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
    // Print1D(h_x1, w, h, level);

    for (int i = 1; i <= repeats; i++)
    {
        // cout << "Step " << i << " of " << repeats << " with count_p = " << count_p << endl;
        
        gradient <<<grid, block>>> (d_delX, d_delY, d_delZ, d_y1, d_y2, d_y3, d_xbar, sigma, w, h, level);
        // cudaMemcpy(h_x1, d_delY, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // cudaMemcpy(h_x2, d_delX, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // cudaMemcpy(h_x3, d_delZ, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // Print3D(h_x1, h_x2, h_x3, w, h, level);
        // Print1D(h_x1, w, h, level);

        set_u_v <<<grid, block>>> (d_u1, d_u2, d_u3, d_v1, d_v2, d_v3, d_delX, d_delY, d_delZ, w, h, level, projections);
        // cudaMemcpy(h_u1, d_u2, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // cudaMemcpy(h_u2, d_u1, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // cudaMemcpy(h_u3, d_u3, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // Print4D(h_u1, h_u2, h_u3, w, h, level, 0);
        // Print3D(h_u1, h_u2, h_u3, w, h, level);
        
        for (int j = 0; j < dykstra; j++)
        {            
            count_p = 1;
            set_u_zero <<<grid, block>>> (d_u1, d_u2, d_u3, w, h, level, projections);
            // Print3D(h_u1, h_u2, h_u3, w, h, level);
            
            // cudaMemcpy(h_u1, d_u2, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
            // cudaMemcpy(h_u2, d_u1, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
            // cudaMemcpy(h_u3, d_u3, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
            // Print4D(h_u1, h_u2, h_u3, w, h, level, 0);
            // parabola y >= p * x^2 + q * x + r.
            project_on_parabola <<<grid, block>>> (d_u1, d_u2, d_u3, d_v1, d_v2, d_v3, d_imgIn, L, lambda, count_p, w, h, level);
            update_v <<<grid, block>>> (d_v1, d_v2, d_v3, d_u1, d_u2, d_u3, w, h, level, count_p);
            // cudaMemcpy(h_u1, d_u2, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
            // cudaMemcpy(h_u2, d_u1, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
            // cudaMemcpy(h_u3, d_u3, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
            // Print4D(h_u1, h_u2, h_u3, w, h, level, count_p);
            count_p++;
            // Print4D(h_u1, h_u2, h_u3, w, h, level, 1);
            // Print4D(h_u1, h_u2, h_u3, w, h, level, 2);
            
            for (int k1 = 0; k1 < level; k1++)
            {
                for (int k2 = k1; k2 < level; k2++)
                {
                    soft_shrinkage <<<grid_iso, block_iso>>> (d_u1, d_u2, d_u3, d_v1, d_v2, d_v3, nu, k1, k2, count_p, w, h, level);
                    update_v <<<grid, block>>> (d_v1, d_v2, d_v3, d_u1, d_u2, d_u3, w, h, level, count_p);
                    // cudaMemcpy(h_u1, d_u2, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
                    // cudaMemcpy(h_u2, d_u1, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
                    // cudaMemcpy(h_u3, d_u3, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
                    // Print4D(h_u1, h_u2, h_u3, w, h, level, count_p);
                    count_p++;
                }
            }

            // for (int k = 1; k < projections; k++)
            // {
            //     update_v <<<grid, block>>> (d_v1, d_v2, d_v3, d_u1, d_u2, d_u3, w, h, level, k);
            // }

            // calc_norm <<<grid, block>>> (d_r1, d_r2, d_r3, d_u1, d_u2, d_u3, w, h, level, projections);
            // cudaMemcpy(h_u1, d_r2, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
            // cudaMemcpy(h_u2, d_r1, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
            // cudaMemcpy(h_u3, d_r3, nbytep, cudaMemcpyDeviceToHost); CUDA_CHECK;
            // Print4D(h_u1, h_u2, h_u3, w, h, level, 0);
            // sum = 0.f;
            // for (int i = 0; i < size; i++)
            // {
            //     sum += pow((h_u1[i] + h_u2[i] + h_u3[i]), 2);
            // }
        }
        // cout << "NORM = " << sqrtf(sum) << endl;
        
        set_y <<<grid, block>>> (d_y1, d_y2, d_y3, d_u1, d_u2, d_u3, w, h, level, projections);
        
        cudaMemcpy(h_x1, d_y1, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(h_x2, d_y2, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(h_x3, d_y3, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;

        sum = 0.f;
        for (int kx = 0; kx < level; kx++)
        {
            for (int ix = 0; ix < h; ix++)
            {
                for (int jx = 0; jx < w; jx++)
                {
                    float x1 = h_x1[jx + w * ix + w * h * kx] - (jx>0 ? h_x1[(jx-1) + w * ix + w * h * kx] : 0.f);
                    float x2 = h_x2[jx + w * ix + w * h * kx] - (ix>0 ? h_x2[jx + w * (ix-1) + w * h * kx] : 0.f);
                    float x3 = h_x3[jx + w * ix + w * h * kx] - (kx>0 ? h_x3[jx + w * ix + w * h * (kx-1)] : 0.f);
                    float d = x1+x2+x3;
                    if (d <= 0) {
                        sum += 1.f;
                        h_r[jx + w * ix + w * h * kx] = 0.f;
                    } else {
                        h_r[jx + w * ix + w * h * kx] = 1.f;
                    }
                }
            }
        }
        printf("%d %f\n", i, sqrtf(sum));

        // Print3D(h_x1, h_x2, h_x3, w, h, level);
        clipping <<<grid, block>>> (d_x, d_xcur, d_y1, d_y2, d_y3, tau, w, h, level);
        // clipping <<<grid, block>>> (d_x, d_xcur, d_delX, d_delY, d_delZ, tau, w, h, level);
        // cudaMemcpy(h_x1, d_xbar, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // Print1D(h_x1, w, h, level);

        extrapolate <<<grid, block>>> (d_xbar, d_xcur, d_x, w, h, level);
        // if (i%20 == 0) {
        //     isosurface <<<grid_iso, block_iso>>> (d_imgOut, d_x, w, h, level);
        //     cudaMemcpy(h_imgOut, d_imgOut, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;
        //     convert_layered_to_mat(mOut, h_imgOut);
        //     showImage("Output", mOut, 100+w+40, 100);
        //     cv::waitKey(0);
        // }
        // Print1D(h_imgOut, w, h, 1);
    }
    // cudaMemcpy(h_x1, d_x, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
    // Print1D(h_x1, w, h, level);

    isosurface <<<grid_iso, block_iso>>> (d_imgOut, d_x, w, h, level);
    // cudaMemcpy(h_imgOut, d_imgOut, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;
    // Print1D(h_imgOut, w, h, 1);

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

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

    cudaFree(d_u1); CUDA_CHECK;
    cudaFree(d_u2); CUDA_CHECK;
    cudaFree(d_u3); CUDA_CHECK;

    cudaFree(d_v1); CUDA_CHECK;
    cudaFree(d_v2); CUDA_CHECK;
    cudaFree(d_v3); CUDA_CHECK;

    // cudaFree(d_r1); CUDA_CHECK;
    // cudaFree(d_r2); CUDA_CHECK;
    // cudaFree(d_r3); CUDA_CHECK;

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
    delete[] h_u1;
    delete[] h_u2;
    delete[] h_u3;
    delete[] h_r;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}