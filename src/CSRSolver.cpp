#include "CSRSolver.h"

CSRSolver::CSRSolver(Geometry& g) : _g(g)
{
	_nnz = countElements();
	_n   = g.nxyz()*_g.ng();
    _rowptr = new int[_n + 1];
	_idx_col = new int[_nnz];
    _idx_diag = new int[_n];
	_a = new double[_nnz];
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
		for (int ige = 0; ige < _g.ng(); ige++)
		{
			for (int idir = NDIRMAX - 1; idir >= 0; --idir)
			{
				int ln = _g.neib(LEFT, idir, l);

				if (ln >= 0) {
					_idx_col[nnz] = ln*_g.ng()+ige;
                    ++nnz;
				}
			}

			//diagonal 
            _idx_diag[l*_g.ng() + ige] = nnz;
			for (int igs = 0; igs < _g.ng(); igs++)
			{
				_idx_col[nnz] = l * _g.ng()+igs;
                ++nnz;
			}

			for (int idir = 0; idir < NDIRMAX; idir++)
			{
				int ln = _g.neib(RIGHT, idir, l);

				if (ln >= 0) {
					_idx_col[nnz] = ln * _g.ng() + ige;;
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

