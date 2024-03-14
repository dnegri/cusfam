#pragma once

#include "Geometry.h"
#include "JacobiBicgSolver.h"

class JacobiBicgSolverCuda : public JacobiBicgSolver {
private:
    double* _crho_dev, *_r0v_dev, *_pts_dev, *_ptt_dev;
    double* _r20_dev, *_r2_dev;

public:
    JacobiBicgSolverCuda(Geometry &g);

    virtual ~JacobiBicgSolverCuda();

    void reset(double *diag, double *cc, double*phi, double *src, double& r20) override;

    void minv(double *cc, double *b, double *x) override;

    void facilu(double *diag, double *cc) override;

    void axb(double *diag, double *cc, double*phi, double *aphi) override;

    void solve(double* diag, double* cc, double& r20, double* phi, double& r2) override;
};


