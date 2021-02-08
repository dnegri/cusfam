//
// Created by JOO IL YOON on 2021/01/22.
//

#include "SuperLUSolver.h"

SuperLUSolver::SuperLUSolver(Geometry &g) : CSRSolver(g) {

    _b = new double[_n]{};
    //dCreate_CompRow_Matrix(&_slu_a, _n, _n, _nnz, _a, _idx_col, _rowptr, SLU_NR, SLU_D, SLU_GE);
    //dCreate_Dense_Matrix(&_slu_b, _n, 1, _b, _n, SLU_DN, SLU_D, SLU_GE);

    //set_default_options(&_slu_opt);

    ////_perm_c = new int[_n]{};
    ////_perm_r = new int[_n]{};
    //if (!(_perm_c = intMalloc(_n))) ABORT("Malloc fails for perm_c[].");
    //if (!(_perm_r = intMalloc(_n))) ABORT("Malloc fails for perm_r[].");

    ///* Initialize the statistics variables. */
    //StatInit(&_slu_stat);

}

SuperLUSolver::~SuperLUSolver() {
    StatFree(&_slu_stat);

    SUPERLU_FREE (_perm_r);
    SUPERLU_FREE (_perm_c);
    Destroy_CompCol_Matrix(&_slu_a);
    Destroy_SuperMatrix_Store(&_slu_b);
    Destroy_SuperNode_Matrix(&_slu_l);
    Destroy_CompCol_Matrix(&_slu_u);

}


void SuperLUSolver::solve(CMFD_VAR *b, double *x) {
    int info;

//    dCreate_CompRow_Matrix(&_slu_a, _n, _n, _nnz, _a, _idx_col, _rowptr, SLU_NR, SLU_D, SLU_GE);
//    dCreate_Dense_Matrix(&_slu_b, _n, 1, b, _n, SLU_DN, SLU_D, SLU_GE);
//
//    set_default_options(&_slu_opt);
//
//    _perm_c = new int[_n]{};
//    _perm_r = new int[_n]{};
//    //if (!(_perm_c = intMalloc(_n))) ABORT("Malloc fails for perm_c[].");
//    //if (!(_perm_r = intMalloc(_n))) ABORT("Malloc fails for perm_r[].");
//
//    /* Initialize the statistics variables. */
//    StatInit(&_slu_stat);
//
//    //memcpy(((NCformat *) _slu_b.Store)->nzval, b, sizeof(double) * _n);
//    dgssv(&_slu_opt, &_slu_a, _perm_c, _perm_r, &_slu_l, &_slu_u, &_slu_b, &_slu_stat, &info);
//    memcpy(x, ((NCformat *) _slu_b.Store)->nzval, sizeof(double) * _n);
//
////    memcpy(x, _slu_b, sizeof(double) * _n);
////    memcpy(((NCformat *) _slu_b.Store)->nzval, _x, sizeof(double) * _n);
////    dgssv(&_slu_opt, &_slu_a, _perm_c, _perm_r, &_slu_l, &_slu_u, &_slu_b, &_slu_stat, &info);

}

void SuperLUSolver::prepare()
{
}

