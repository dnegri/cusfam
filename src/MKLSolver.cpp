#include <mkl.h>
#include "MKLSolver.h"

MKLSolver::MKLSolver(Geometry& g) : CSRSolver(g)
{
}

MKLSolver::~MKLSolver()
{

}

void MKLSolver::solve(double* b, double* x)
{
	char transa = 'N';
	mkl_cspblas_dcsrgemv(&transa, &nnz(), _a, indexRow(), indexCol(), b, x);
}
