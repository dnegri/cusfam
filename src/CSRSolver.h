#pragma once
#include "pch.h"
#include "Geometry.h"

class CSRSolver : public Managed
{
protected:
    int _n;
	int _nnz;
	int* _rowptr;
	int* _idx_col;
    int * _idx_diag;
	int* _fac;
    double* _a;
	Geometry _g;

public:
	CSRSolver(Geometry& g);
	virtual ~CSRSolver();

	virtual void solve(CMFD_VAR* b, double* x) = 0;
	virtual void prepare() = 0;
	const int* rowptr() { return _rowptr; };
	const int& rowptr(const int& idx_row) { return _rowptr[idx_row]; };
	const int& idxdiag(const int& idx_row) { return _idx_diag[idx_row]; };
	const int* indexCol() { return _idx_col; };
	const int& nnz() { return _nnz; };
	int& fac(const int& lr, const int& idir, const int& l) { return _fac[l*NDIRMAX*LR + idir*LR + lr]; };

	double& a(const int& index) { return _a[index]; };

private:
	void initialize();
	int  countElements();

};

