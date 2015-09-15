#include "Vector.h"
#include <cmath>

#ifndef __ALGEBRA_H__
#define __ALGEBRA_H__

template<class F>
class Algebra
{
public:
	Algebra() {}
	~Algebra() {}

	void Image2Vector(primaldual::Vector<F>& dst, Image<F>& src) {
		if (!(src.Height() == dst.Height() && src.Width() == dst.Width())) {
			cout << "ERROR 14 (Image2Vector): Height and width of image and vector do not match." << endl;
		} else {
			for (int k = 0; k < dst.Level(); k++) {
				for (int i = 0; i < dst.Height(); i++) {
					for (int j = 0; j < dst.Width(); j++) {
						if (src.Get(i, j) > ((F)k/(F)dst.Level())) {
							dst.Set(i, j, k, 0, 1.0);
						} else {
							dst.Set(i, j, k, 0, 0.0);
						}
					}
				}
			}
		}
	}

	void AddVector(primaldual::Vector<F>& dst, primaldual::Vector<F>& src1, primaldual::Vector<F>& src2, F factor1, F factor2) {
		if (!(dst.EqualProperties(src1) && dst.EqualProperties(src2))) {
			cout << "ERROR 01 (AddVector): Height, Width, Level and/or Dimension do not match for used vectors." << endl;
		} else {
			for (int c = 0; c < src1.Dimension(); c++)
				for (int k = 0; k < src1.Level(); k++)
					for (int i = 0; i < src1.Height(); i++)
						for (int j = 0; j < src1.Width(); j++)
							dst.Set(i, j, k, c, factor1 * src1.Get(i, j, k, c) + factor2 * src2.Get(i, j, k, c));
		}
	}

	void ScaleVector(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, F factor) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 02 (ScaleVector): Height, Width, Level and/or Dimension do not match for used vectors." << endl;
		} else {
			for (int c = 0; c < src.Dimension(); c++)
				for (int k = 0; k < src.Level(); k++)
					for (int i = 0; i < src.Height(); i++)
						for (int j = 0; j < src.Width(); j++)
							dst.Set(i, j, k, c, factor * src.Get(i, j, k, c));
		}
	}
};

#endif //__ALGEBRA_H__