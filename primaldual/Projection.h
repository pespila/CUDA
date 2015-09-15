#include "Algebra.h"
#include "Vector.h"
#include <cmath>

#ifndef __PROJECTION_H__
#define __PROJECTION_H__

template<class F>
class Projection
{
private:
	int level;
	int semilevel;
	int constraints;

	primaldual::Vector<F>* u;
	primaldual::Vector<F>* v;

	void Reset() {
		for (int k = 0; k < constraints; k++) {
			for (int c = 0; c < 3; c++) {
				u[k].Set(0, c, 0.0);
				v[k].Set(0, c, 0.0);
			}
		}
	}
public:
	Projection() {}
	Projection(int level) {
		this->level = level;
		this->semilevel = level * (level + 1) / 2;
		this->constraints = level + semilevel + 1;
		this->u = new primaldual::Vector<F>[constraints];
		this->v = new primaldual::Vector<F>[constraints];
		for (int i = 0; i < constraints; i++) {
			u[i] = primaldual::Vector<F>(1, 3, 0.0);
			v[i] = primaldual::Vector<F>(1, 3, 0.0);
		}
	}
	~Projection() {
		delete[] this->u;
		delete[] this->v;
	}
	void TruncationOperation(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, int i, int j, int k) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 08 (TruncationOperation): Height, Width, Level and/or Dimension do not match for used vectors." << endl;
		} else {
			if (k == 0) {
				dst.Set(i, j, k, 0, 1.0);
			} else if (k == level - 1) {
				dst.Set(i, j, k, 0, 0.0);
			} else {
				dst.Set(i, j, k, 0, fmin(1.0, fmax(0.0, src.Get(i, j, k, 0))));
			}
		}
	}
	void ProjectOnParabola(primaldual::Vector<F>& dst, F y1, F y2, F y3, F f, F L, F lambda, F k) {
		if (dst.Size() != 3) {
			cout << "ERROR 09 (OnParabola): Wrong size of vector dst!" << endl;
		} else {
			F y = y3 + lambda * (pow(k / L - f, 2));
			F norm = sqrt(pow(y1, 2) + pow(y2, 2));
			F parabola = 0.25 * pow(norm, 2);
			if (y >= parabola) {
				dst.Set(0, 0, y1);
				dst.Set(0, 1, y2);
				dst.Set(0, 2, y3);
			} else {
				F v = 0;
				F a = 2.0 * 0.25 * norm;
				F b = 2.0 / 3.0 * (1.0 - 2.0 * 0.25 * y);
				F d = b < 0 ? (a - pow(sqrt(-b), 3)) * (a + pow(sqrt(-b), 3)) : a * a + b * b * b;
				F c = pow((a + sqrt(d)), 1.0 / 3.0);
				if (d >= 0) {
					v = c != 0 ? c - b / c : 0.0;
				} else {
					v = 2.0 * sqrt(-b) * cos((1.0 / 3.0) * acos(a / (pow(sqrt(-b), 3))));
				}
				dst.Set(0, 0, norm != 0 ? (v / (2.0 * 0.25)) * y1 / norm : 0.0);
				dst.Set(0, 1, norm != 0 ? (v / (2.0 * 0.25)) * y2 / norm : 0.0);
				norm = dst.EuclideanNorm(2);
				parabola = 0.25 * pow(norm, 2);
				dst.Set(0, 2, parabola - lambda * (pow(k / L - f, 2)));
			}
		}
	}
	void L2(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, F bound) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 06 (L2): Height, Width, Level and/or Dimension do not match for used vectors." << endl;
		} else {
			F norm = src.EuclideanNorm();
			for (int i = 0; i < src.Size(); i++) {
				if (norm <= bound) {
					dst.Set(0, i, src.Get(0, i));
				} else {
					dst.Set(0, i, bound * (src.Get(0, i) / norm));
				}
			}
		}
	}
	void SoftShrinkage(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, int i, int j, int k, int k1, int k2, F nu) {
		if (dst.Size() != 3) {
			cout << "ERROR 10 (SoftShrinkage): Size of dst does not match!" << endl;
		} else {
			F K = (F)(k2 - k1 + 1);
			primaldual::Vector<F> s(1, 2, 0.0);
			primaldual::Vector<F> s0(1, 2, 0.0);
			for (int l = k1; l <= k2; l++) {
				for (int c = 0; c < 2; c++) {
					s0.Set(0, c, s0.Get(0, c) + src.Get(i, j, l, c));
				}
			}
			L2(s, s0, nu);
			for (int c = 0; c < 2; c++) {
				if (k >= k1 && k <= k2) {
					dst.Set(0, c, src.Get(i, j, k, c) + ((s.Get(0, c) - s0.Get(0, c)) / K));
				} else {
					dst.Set(0, c, src.Get(i, j, k, c));
				}
			}
		}
	}
	void Dykstra(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, Image<F>& img, Algebra<F>& linalg, F nu, F L, F lambda, int i, int j, int k, int dykstra_max_iter) {
		Reset();
		F norm = 0;
		for (int c = 0; c < 3; c++) {
			u[constraints-1].Set(0, c, src.Get(i, j, k, c));
		}
		for (int m = 0; m < dykstra_max_iter; m++) {
			norm = 0;
			for (int c = 0; c < 3; c++) {
				u[0].Set(0, c, u[constraints-1].Get(0, c));
			}
			for (int l = 1; l < constraints; l++) {
				for (int c = 0; c < 3; c++) {
					u[l].Set(0, c, u[l-1].Get(0, c) - v[l].Get(0, c));
				}
				if (l <= level) {
					ProjectOnParabola(u[l], src.Get(i, j, k, 0), src.Get(i, j, k, 1), u[l].Get(0, 2), img.Get(i, j), L, lambda, k);
				} else {
					for (int k1 = 0; k1 < level; k1++) {
						for (int k2 = k1; k2 < level; k2++) {
							SoftShrinkage(u[l], src, i, j, k, k1, k2, nu);
						}
					}
				}
			}
			for (int l = 1; l < constraints; l++) {
				for (int c = 0; c < 3; c++) {
					v[l].Set(0, c, u[l].Get(0, c) - (u[l-1].Get(0, c) - v[l].Get(0, c)));
				}
			}
			for (int c = 0; c < 3; c++) {
				norm += pow(u[0].Get(0, c) - u[constraints-1].Get(0, c), 2);
			}
			norm = sqrt(norm);
			if (norm < pow(10, -6)) {
				break;
			}
		}
		for (int c = 0; c < 3; c++) {
			dst.Set(i, j, k, c, u[constraints-1].Get(0, c));
		}
	}
};

#endif //__PROJECTION_H__