#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

#ifndef __VECTOR_H__
#define __VECTOR_H__

namespace primaldual {

	template<class F>
	class Vector
	{
	private:
		int height;
		int width;
		int level;
		int dimension;
		vector<F> v;

	public:
		Vector() {}
		Vector(int height, int width, F value) {
			this->v.resize(height*width, value);
			this->height = height;
			this->width = width;
			this->level = 0;
			this->dimension = 0;
		}
		Vector(int height, int width, int level, int dimension, F value) {
			this->v.resize(height*width*level*dimension, value);
			this->height = height;
			this->width = width;
			this->level = level;
			this->dimension = dimension;
		}
		~Vector() {this->v.clear();}

		F Get(int i, int j, int k, int l) {return this->v[j + i * width + k * height * width + l * height * width * level];}
		F Get(int i, int j) {return this->v[j + i * width];}
		void Set(int i, int j, int k, int l, F value) {this->v[j + i * width + k * height * width + l * height * width * level] = value;}
		void Set(int i, int j, F value) {this->v[j + i * width] = value;}

		int Height() {return this->height;}
		int Width() {return this->width;}
		int Level() {return this->level;}
		int Dimension() {return this->dimension;}
		int Size() {return this->v.size();}

		int EqualProperties(Vector<F>& src) {
			return Height() == src.Height() && Width() == src.Width() && Level() == src.Level() && Dimension() == src.Dimension() ? 1 : 0;
		}
		F EuclideanNorm(int s = 0) {
			F norm = 0.0;
			int size = s == 0 ? Size() : s;
			for (int i = 0; i < size; i++)
				norm += pow(Get(0, i, 0, 0), 2);
			return sqrt(norm);
		}

		void Print() {
			for (int k = 0; k < Level(); k++) {
				cout << "_____ Level: " << k << " _____" << endl;
				for (int i = 0; i < Height(); i++) {
					for (int j = 0; j < Width(); j++) {
						cout << "( ";
						for (int c = 0; c < Dimension(); c++) {
							// printf("%g ", Get(i, j, k, c));
							cout << Get(i, j, k, c) << " ";
						}
						cout << ")" << "  ";
					}
					cout << endl;
				}
				cout << "_____          _____" << endl << endl;
			}
		}
	};

}

#endif //__VECTOR_H__