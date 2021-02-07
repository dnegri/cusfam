//
// Created by JOO IL YOON on 2021/02/06.
//

#include "GinkgoSolver.h"

GinkgoSolver::GinkgoSolver(Geometry& g) : _g(g) {

}

GinkgoSolver::~GinkgoSolver() {

}

void GinkgoSolver::initialize() {
//    for (int l = 0; l < _g.nxyz(); l++)
//    {
//        for (int idir = NDIRMAX - 1; idir >= 0; --idir)
//        {
//            int ln = _g.neib(LEFT, idir, l);
//
//            if (ln >= 0) {
//                idx_col_onerow[idir] = ln;
//            }
//        }
//
//        for (int idir = 0; idir < NDIRMAX; idir++)
//        {
//            int ln = _g.neib(RIGHT, idir, l);
//
//            if (ln >= 0 && ln != l) {
//                idx_col_onerow[NDIRMAX+idir] = ln;
//            }
//        }
//
//        for (int idirl = 0; idirl < NDIRMAX; idirl++)
//        {
//            for (int idirr = 0; idirr < NDIRMAX; idirr++)
//            {
//                if (idx_col_onerow[idirl] != -1 && idx_col_onerow[idirl] == idx_col_onerow[NDIRMAX + idirr]) {
//                    fac(RIGHT, idirr, l) = fac(RIGHT, idirr, l) + 1;
//                    fac(LEFT, idirl, l) = 0;
//                    break;
//                }
//            }
//        }
//
//        for (int ige = 0; ige < _g.ng(); ige++)
//        {
//
//
//            for (int idir = NDIRMAX - 1; idir >= 0; --idir)
//            {
//                int ln = _g.neib(LEFT, idir, l);
//
//                if(ln >= 0 && ln != l && fac(LEFT, idir, l) != 0) {
//                    _idx_col[nnz] = ln * _g.ng() + ige;
//                    ++nnz;
//                }
//            }
//
//            //diagonal
//            _idx_diag[l * _g.ng() + ige] = nnz;
//            for (int igs = 0; igs < _g.ng(); igs++)
//            {
//                _idx_col[nnz] = l * _g.ng()+igs;
//                ++nnz;
//            }
//
//
//            for (int idir = 0; idir < NDIRMAX; idir++)
//            {
//                int ln = _g.neib(RIGHT, idir, l);
//
//                if (ln >= 0 && ln != l && fac(RIGHT, idir, l) != 0) {
//                    _idx_col[nnz] = ln * _g.ng() + ige;
//                    ++nnz;
//                }
//            }
//
//            _rowptr[l * _g.ng() + ige + 1] = nnz;
//        }
//    }
}
