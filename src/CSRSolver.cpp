#include "CSRSolver.h"

CSRSolver::CSRSolver(Geometry& g) : _g(g)
{
	_nnz = countElements();
	_idx_row = new int[(g.nxyz()*_g.ng())+1];
	_idx_col = new int[_nnz];
	initialize();
}

CSRSolver::~CSRSolver()
{
}

void CSRSolver::initialize()
{
	_idx_row[0] = 0;

	for (size_t l = 0; l < _g.nxyz(); l++)
	{
		for (size_t ige = 0; ige < _g.ng(); ige++)
		{
			int nel = 0;
			for (size_t idir = NDIRMAX - 1; idir >= 0; --idir)
			{
				int ln = _g.neib(LEFT, idir, l);

				if (ln >= 0) {
					++nel;
					*_idx_col++ = ln*_g.ng()+ige;
				}
			}

			//diagonal 
			for (size_t igs = 0; igs < _g.ng(); igs++)
			{
				++nel;
				*_idx_col++ = l * _g.ng()+igs;
			}

			for (size_t idir = 0; idir < NDIRMAX; idir++)
			{
				int ln = _g.neib(RIGHT, idir, l);

				if (ln >= 0) {
					++nel;
					*_idx_col++ = ln * _g.ng() + ige;;
				}
			}

			_idx_row[l*_g.ng() + ige + 1] = nel;
		}
	}
}

int CSRSolver::countElements()
{
	int cnt = _g.nxyz()*_g.ng2(); //the number of diagonal elements

	for (size_t l = 0; l < _g.nxyz(); l++)
	{
		for (size_t idir = 0; idir < NDIRMAX; idir++)
		{
			for (size_t lr = 0; lr < LR; lr++)
			{
				int ln = _g.neib(lr, idir, l);

				if (ln >= 0) {
					cnt += cnt*_g.ng();
				}
			}
		}
	}

	return cnt;
}
