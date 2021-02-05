#include "CSRSolver.h"

CSRSolver::CSRSolver(Geometry& g) : _g(g)
{
	_nnz = countElements();
	_n   = g.nxyz()*_g.ng();
    _rowptr = new int[_n + 1];
	_idx_col = new int[_nnz];
    _idx_diag = new int[_n];
	_a = new double[_nnz];
	_fac = new int[LR*NDIRMAX*g.nxyz()];
	std::fill_n(_fac, LR * NDIRMAX * g.nxyz(), 1);
	initialize();
}

CSRSolver::~CSRSolver()
{
}

void CSRSolver::initialize()
{
    _rowptr[0] = 0;

    int nnz = 0;
	for (int l = 0; l < _g.nxyz(); l++)
	{
		int idx_col_onerow[NDIRMAX * LR];
		std::fill_n(idx_col_onerow, NDIRMAX * LR, -1);

		for (int idir = NDIRMAX - 1; idir >= 0; --idir)
		{
			int ln = _g.neib(LEFT, idir, l);

			if (ln >= 0 && ln != l) {
				idx_col_onerow[idir] = ln;
			}
		}

		for (int idir = 0; idir < NDIRMAX; idir++)
		{
			int ln = _g.neib(RIGHT, idir, l);

			if (ln >= 0 && ln != l) {
				idx_col_onerow[NDIRMAX+idir] = ln;
			}
		}

		for (int idirl = 0; idirl < NDIRMAX; idirl++)
		{
			for (int idirr = 0; idirr < NDIRMAX; idirr++)
			{
				if (idx_col_onerow[idirl] != -1 && idx_col_onerow[idirl] == idx_col_onerow[NDIRMAX + idirr]) {
					fac(RIGHT, idirr, l) = fac(RIGHT, idirr, l) + 1;
					fac(LEFT, idirl, l) = 0;
					break;
				}
			}
		}

		for (int ige = 0; ige < _g.ng(); ige++)
		{


			for (int idir = NDIRMAX - 1; idir >= 0; --idir)
			{
				int ln = _g.neib(LEFT, idir, l);

				if(ln >= 0 && ln != l && fac(LEFT, idir, l) != 0) {
					_idx_col[nnz] = ln * _g.ng() + ige;
					++nnz;
				}
			}

			//diagonal 
			_idx_diag[l * _g.ng() + ige] = nnz;
			for (int igs = 0; igs < _g.ng(); igs++)
			{
				_idx_col[nnz] = l * _g.ng()+igs;
				++nnz;
			}


			for (int idir = 0; idir < NDIRMAX; idir++)
			{
				int ln = _g.neib(RIGHT, idir, l);

				if (ln >= 0 && ln != l && fac(RIGHT, idir, l) != 0) {
					_idx_col[nnz] = ln * _g.ng() + ige;
					++nnz;
				}
			}

            _rowptr[l * _g.ng() + ige + 1] = nnz;
		}
	}
}

int CSRSolver::countElements()
{
	int nnz = _g.nxyz()*_g.ng2(); //the number of diagonal elements

	for (int l = 0; l < _g.nxyz(); l++)
	{
		for (int idir = 0; idir < NDIRMAX; idir++)
		{
			for (int lr = 0; lr < LR; lr++)
			{
				int ln = _g.neib(lr, idir, l);

				if (ln >= 0) {
					nnz += _g.ng();
				}
			}
		}
	}

	return nnz;
}

