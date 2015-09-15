#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

#ifndef __IMAGE_H__
#define __IMAGE_H__

template<class F>
class Image
{
private:
	vector<F> v;
	int height;
	int width;
public:
	Image():height(0), width(0) {}
	Image(int height, int width, F value = (F)0) {
		v.resize(height * width, value);
		this->height = height;
		this->width = width;
	}
	~Image() {v.clear();}

	F Get(int i, int j) {return this->v[j + i * width];}
	void Set(int i, int j, F value) {this->v[j + i * width] = value;}
	
	int Height() {return this->height;}
	int Width() {return this->width;}
	int Size() {return this->v.size();}
	
	void Print() {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				cout << Get(i, j) << " ";
			}
			cout << endl;
		}
	}

	void Read(const string filename) {
		Mat img = imread(filename, 0); // force gray scale
	    this->width = img.cols;
	    this->height = img.rows;
	    this->v.resize(height * width, 0.0);
	    for (int i = 0; i < height; i++) {
	        for (int j = 0; j < width; j++) {
	            this->v[j + i * width] = img.at<uchar>(i, j);
	        }
	    }
	}

	void Write(const string filename) {
		Mat img(height, width, CV_8UC1);
	    for (int i = 0; i < height; i++) {
	        for (int j = 0; j < width; j++) {
	            img.at<uchar>(i, j) = (int)v[j + i * width];
	        }
	    }
	    imwrite(filename, img);
	}
};

#endif //__IMAGE_H__