#pragma once
#include <mkl.h>
#include "CSRSolver.h"

class MKLSolver : public CSRSolver
{
private:
	void* pt[64];
	MKL_INT*	iparam;
	MKL_INT* idum;

	MKL_INT _nrhs = 1;
	MKL_INT _maxfct = 1;
	MKL_INT _mnum = 1;
	MKL_INT _msglvl = 0; // print statistical information
	MKL_INT _mtype = 11; // nonsymmetric

public:
	MKLSolver(Geometry& g);
	virtual ~MKLSolver();

	void solve(double* b, double* x);
	void prepare();
};

