#pragma once
#include "pch.h"
#include "Geometry.h"

class CSRSolver : public Managed
{
protected:
	int _nnz;
	int* _idx_row;
	int* _idx_col;

	double* _a;
	Geometry _g;

public:
	CSRSolver(Geometry& g);
	virtual ~CSRSolver();

	virtual void solve(double* b, double* x) = 0;
	const int* indexRow() { return _idx_row; };
	const int& indexRow(const int& idx_row) { return _idx_row[idx_row]; };
	const int* indexCol() { return _idx_col; };
	const int& nnz() { return _nnz; };

	double& a(const int& index) { return _a[index]; };

private:
	void initialize();
	int  countElements();

};

