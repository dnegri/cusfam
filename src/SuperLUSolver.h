#pragma once

#include "slu_ddefs.h"
#include "CSRSolver.h"

class SuperLUSolver : public CSRSolver {
private:
    SuperMatrix _slu_a;
    SuperMatrix _slu_l;      /* factor L */
    SuperMatrix _slu_u;      /* factor U */
    SuperMatrix _slu_b;
    superlu_options_t _slu_opt;
    SuperLUStat_t _slu_stat;

    int *_perm_c; /* column permutation vector */
    int *_perm_r; /* row permutations from partial pivoting */

    double * _b;

public:
    virtual ~SuperLUSolver();

public:
    SuperLUSolver(Geometry &g);

public:
    void solve(double *b, double *x) override;
    void prepare() override;

};


