#pragma once

#include "Geometry.h"
#include "JacobiBicgSolver.h"

class JacobiBicgSolverCuda : public JacobiBicgSolver {
private:
    CMFD_VAR* _crho_dev, *_r0v_dev, *_pts_dev, *_ptt_dev, * _r2_dev;

public:
    JacobiBicgSolverCuda(Geometry &g);

    virtual ~JacobiBicgSolverCuda();

    void reset(CMFD_VAR *diag, CMFD_VAR *cc, SOL_VAR*phi, CMFD_VAR *src, CMFD_VAR& r20);

    void minv(CMFD_VAR *cc, CMFD_VAR *b, SOL_VAR *x);

    void facilu(CMFD_VAR *diag, CMFD_VAR *cc);

    void axb(CMFD_VAR *diag, CMFD_VAR *cc, SOL_VAR*phi, CMFD_VAR *aphi);

    void solve(CMFD_VAR* diag, CMFD_VAR* cc, CMFD_VAR& r20, SOL_VAR* phi, double& r2);
};


