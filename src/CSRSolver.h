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
public:
    const int& idxdiag(const int& idx_row) {
        return _idx_diag[idx_row];
    };

protected:


    double* _a;
	Geometry _g;

public:
	CSRSolver(Geometry& g);
	virtual ~CSRSolver();

	virtual void solve(double* b, double* x) = 0;
	const int* rowptr() { return _rowptr; };
	const int& rowptr(const int& idx_row) { return _rowptr[idx_row]; };
	const int* indexCol() { return _idx_col; };
	const int& nnz() { return _nnz; };

	double& a(const int& index) { return _a[index]; };

private:
	void initialize();
	int  countElements();

};

