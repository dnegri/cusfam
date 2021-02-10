#pragma once

#include "Geometry.h"
#include "JacobiBicgSolver.h"

class JacobiBicgSolverCuda : public JacobiBicgSolver {

public:
    JacobiBicgSolverCuda(Geometry &g);

    virtual ~JacobiBicgSolverCuda();

    void reset(CMFD_VAR *diag, CMFD_VAR *cc, double *phi, CMFD_VAR *src, CMFD_VAR& r20);

    void minv(CMFD_VAR *cc, CMFD_VAR *b, double *x);

    void facilu(CMFD_VAR *diag, CMFD_VAR *cc);

    void axb(CMFD_VAR *diag, CMFD_VAR *cc, double *phi, CMFD_VAR *aphi);

    void solve(CMFD_VAR* diag, CMFD_VAR* cc, CMFD_VAR& r20, double* phi, double& r2);
};


