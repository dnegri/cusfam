#pragma once
#include "CSRSolver.h"

class MKLSolver : public CSRSolver
{
public:
	MKLSolver(Geometry& g);
	virtual ~MKLSolver();

	void solve(double* b, double* x);
};

