//
// Created by JOO IL YOON on 2021/01/30.
//
#include "SSORSolver.h"
#include "mat2g.h"
#include "myblas.h"

#define diag(igs, ige, l)   diag[(l) * _g->ng2() + (ige) * _g->ng() + (igs)]
#define cc(lr, idir, ig, l) cc[(l) * _g->ng() * NDIRMAX * LR + (ig) * NDIRMAX * LR + (idir) * LR + (lr)]
#define src(ig, l)          src[(l) * _g->ng() + (ig)]
#define aflux(ig, l)        aflux[(l) * _g->ng() + (ig)]
#define b(ig, l)            b[(l) * _g->ng() + (ig)]
#define x(ig, l)            x[(l) * _g->ng() + (ig)]
#define flux(ig, l)         flux[(l) * _g->ng() + (ig)]

SSORSolver::SSORSolver(Geometry& g) {

    _g = &g;

    _rhoJacobi = 0.99;
    _rhoGS     = pow(_rhoJacobi, 2);
    _omega     = 2 * (1 - sqrt(1 - _rhoGS)) / _rhoGS;

    _delinv = new double[_g->ng2() * _g->nxyz()]{};
}

SSORSolver::~SSORSolver() {
    delete _delinv;
}

double SSORSolver::reset(const int& l, double* diag, double* cc, double* flux, double* src) {
    double r = 0.0;
    for (int ig = 0; ig < _g->ng(); ig++) {
        double aflux = axb(ig, l, diag, cc, flux);
        double vr    = src(ig, l) - aflux;
        r += vr * vr;
    }

    return r;
}

void SSORSolver::reset(double* diag, double* cc, double* flux, double* src, double& r20) {
    _src = src;

    //     r20 = 0;
    // #pragma omp parallel for reduction(+ : r20)
    //     for (int l = 0; l < _g->nxyz(); ++l) {
    //         r20 += reset(l, diag, cc, flux, src);
    //     }

    //     r20 = sqrt(r20);

#pragma omp parallel for
    for (int l = 0; l < _g->nxyz(); ++l) {
        invmat2g(&diag(0, 0, l), &delinv(0, 0, l));
    }
}

void SSORSolver::minv(double* cc, double* b, double* x, double& errl2) {

    errl2        = 0.0;
    double flux2 = 0.0;
    for (int rb = 0; rb < 2; ++rb) {
#pragma omp parallel for reduction(+ : errl2, flux2)
        for (int k = 0; k < _g->nz(); ++k) {
            for (int j = 0; j < _g->ny(); ++j) {
                for (int i = 0; i < _g->nx(); ++i) {
                    if ((i + j + k + 1) % 2 == rb) continue;
                    if (i < _g->nxs(j) || i >= _g->nxe(j)) continue;

                    int l2d = _g->ijtol(i, j);
                    int l   = k * _g->nxy() + l2d;

                    minv(l, cc, b, x, errl2);
                    flux2 = flux2 + x(1, l) * x(1, l);
                }
            }
        }
    }

    errl2 = errl2 / flux2;
}

void SSORSolver::minv(const int& l, double* cc, double* b, double* x, double& errl2) {

    double aflux[2]{};
    for (int ig = 0; ig < _g->ng(); ++ig) {
        aflux[ig] = b(ig, l);
        for (int idir = 0; idir < NDIRMAX; ++idir) {
            for (int lr = 0; lr < LR; ++lr) {
                int ln = _g->neib(lr, idir, l);
                if (ln != -1)
                    aflux[ig] -= cc(lr, idir, ig, l) * x(ig, ln);
            }
        }
    }

    double xnew[2]{};
    matxvec2g(&delinv(0, 0, l), aflux, xnew);

    for (int ig = 0; ig < _g->ng(); ++ig) {
        aflux[ig] = x(ig, l);
        x(ig, l)  = (1 - _omega) * x(ig, l) + _omega * xnew[ig];
    }
    errl2 += pow((x(1, l) - aflux[1]), 2);
}

void SSORSolver::facilu(double* diag, double* cc) {
}
void SSORSolver::facilu(const int& l, double* diag, double* cc) {
}

void SSORSolver::solve(double* diag, double* cc, double& r20, double* flux, double& r2) {

    r2 = 0.0;
    minv(cc, _src, flux, r2);
    // r2 = myblas::dot(_g->nxyz() * _g->ng(), flux, flux);

    // r2 = errl2 / r2;

    // reset(diag, cc, flux, _src, r2);

    // if (r20 != 0.0) {
    //     r2 = sqrt(r2) / r20;
    // }
}

void SSORSolver::axb(double* diag, double* cc, double* flux, double* aflux) {

#pragma omp parallel for
    for (int l = 0; l < _g->nxyz(); ++l) {
        for (int ig = 0; ig < _g->ng(); ++ig) {
            aflux(ig, l) = axb(ig, l, diag, cc, flux);
        }
    }
}

double SSORSolver::axb(const int& ig, const int& l, double* diag, double* cc, double* flux) {

    double ab = 0.0;

    for (int igs = 0; igs < _g->ng(); ++igs) {
        ab += diag(igs, ig, l) * flux(igs, l);
    }

    for (int idir = 0; idir < NDIRMAX; ++idir) {
        for (int lr = 0; lr < LR; ++lr) {
            int ln = _g->neib(lr, idir, l);
            if (ln != -1)
                ab += cc(lr, idir, ig, l) * flux(ig, ln);
        }
    }

    return ab;
}

#undef diag
#undef cc
#undef src
#undef aflux
#undef b
#undef x
#undef flux
#undef vr
#undef vr0
#undef vp
#undef vv
#undef delinv
