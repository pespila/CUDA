#include <iostream>
#include <vector>

using namespace std;

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
};

#endif //__IMAGE_H__