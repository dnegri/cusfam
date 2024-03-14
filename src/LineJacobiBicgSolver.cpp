//
// Created by JOO IL YOON on 2021/01/30.
//
#include "LineJacobiBicgSolver.h"
#include "mat2g.h"
#include "myblas.h"

#define diag(igs, ige, l)   diag[(l) * _g->ng2() + (ige) * _g->ng() + (igs)]
#define cc(lr, idir, ig, l) cc[(l) * _g->ng() * NDIRMAX * LR + (ig) * NDIRMAX * LR + (idir) * LR + (lr)]
#define src(ig, l)          src[(l) * _g->ng() + (ig)]
#define aflux(ig, l)        aflux[(l) * _g->ng() + (ig)]
#define b(ig, l)            b[(l) * _g->ng() + (ig)]
#define x(ig, l)            x[(l) * _g->ng() + (ig)]
#define flux(ig, l)         flux[(l) * _g->ng() + (ig)]

LineJacobiBicgSolver::LineJacobiBicgSolver(Geometry& g) {

    _g = &g;

    _calpha = 0.0;
    _cbeta  = 0.0;
    _crho   = 0.0;
    _comega = 0.0;

    _vz = new double[_g->ng() * _g->nxyz()]{};
    _vy = new double[_g->ng() * _g->nxyz()]{};

    _vr     = new double[_g->ng() * _g->nxyz()]{};
    _vr0    = new double[_g->ng() * _g->nxyz()]{};
    _vp     = new double[_g->ng() * _g->nxyz()]{};
    _vv     = new double[_g->ng() * _g->nxyz()]{};
    _vs     = new double[_g->ng() * _g->nxyz()]{};
    _vt     = new double[_g->ng() * _g->nxyz()]{};
    _delinv = new double[_g->ng2() * _g->nxyz()]{};
    _delcc  = new double[_g->ng2() * _g->nxyz()]{};
}

LineJacobiBicgSolver::~LineJacobiBicgSolver() {
    delete _vr;
    delete _vr0;
    delete _vp;
    delete _vv;
    delete _vs;
    delete _vt;
    delete _vy;
    delete _vz;
    delete _delinv;
    delete _delcc;
}

double LineJacobiBicgSolver::reset(const int& l, double* diag, double* cc, double* flux, double* src) {

    double r = 0.0;
    for (int ig = 0; ig < _g->ng(); ig++) {
        double aflux = axb(ig, l, diag, cc, flux);
        vr(ig, l)    = src(ig, l) - aflux;
        vr0(ig, l)   = vr(ig, l);
        vp(ig, l)    = 0.0;
        vv(ig, l)    = 0.0;
        r += vr(ig, l) * vr(ig, l);
    }

    return r;
}

void LineJacobiBicgSolver::reset(double* diag, double* cc, double* flux, double* src, double& r20) {

    _calpha = 1;
    _crho   = 1;
    _comega = 1;

    r20 = 0;

#pragma omp parallel for reduction(+ : r20)
    for (int l = 0; l < _g->nxyz(); ++l) {
        r20 += reset(l, diag, cc, flux, src);
    }

    r20 = sqrt(r20);
}

void LineJacobiBicgSolver::minv(double* cc, double* b, double* x) {

#pragma omp parallel for
    for (int k = 0; k < _g->nz(); ++k) {
        for (int j = 0; j < _g->ny(); ++j) {
            minv(j, k, cc, b, x);
        }
    }
}

void LineJacobiBicgSolver::minv(const int& j, const int& k, double* cc, double* b, double* x) {

    int l2d = _g->ijtol(_g->nxs(j), j);
    int l   = l2d + k * _g->nxy();

    x(0, l) = b(0, l);
    x(1, l) = b(1, l);

    for (int i = _g->nxs(j) + 1; i < _g->nxe(j); i++) {
        l2d     = _g->ijtol(i, j);
        l       = l2d + k * _g->nxy();
        x(0, l) = b(0, l) - (delcc(0, 0, l) * x(0, l - 1) + delcc(1, 0, l) * x(1, l - 1));
        x(1, l) = b(1, l) - (delcc(0, 1, l) * x(0, l - 1) + delcc(1, 1, l) * x(1, l - 1));
    }

    l2d = _g->ijtol(_g->nxe(j) - 1, j);
    l   = l2d + k * _g->nxy();

    double y[2];
    y[0] = b(0, l);
    y[1] = b(1, l);

    x(0, l) = delinv(0, 0, l) * y[0] + delinv(1, 0, l) * y[1];
    x(1, l) = delinv(0, 1, l) * y[0] + delinv(1, 1, l) * y[1];

    for (int i = _g->nxe(j) - 2; i >= _g->nxs(j); i--) {
        l2d = _g->ijtol(i, j);
        l   = l2d + k * _g->nxy();

        x(0, l) = x(0, l) - cc(RIGHT, XDIR, 0, l) * x(0, l + 1);
        x(1, l) = x(1, l) - cc(RIGHT, XDIR, 1, l) * x(1, l + 1);

        y[0] = x(0, l);
        y[1] = x(1, l);

        x(0, l) = delinv(0, 0, l) * y[0] + delinv(1, 0, l) * y[1];
        x(1, l) = delinv(0, 1, l) * y[0] + delinv(1, 1, l) * y[1];
    }

    // for (int i = _g->nxs(j); i < _g->nxe(j); i++) {
    //     int l2d = _g->ijtol(i, j);
    //     int l   = l2d + k * _g->nxy();

    //     x(0, l) = delinv(0, 0, l) * b(0, l) + delinv(1, 0, l) * b(0, l);
    //     x(1, l) = delinv(0, 1, l) * b(1, l) + delinv(1, 1, l) * b(1, l);
    // }
}

void LineJacobiBicgSolver::facilu(double* diag, double* cc) {

#pragma omp parallel for
    for (int k = 0; k < _g->nz(); ++k) {
        for (int j = 0; j < _g->ny(); ++j) {
            facilu(j, k, diag, cc);
        }
    }
}
void LineJacobiBicgSolver::facilu(const int& j, const int& k, double* diag, double* cc) {

    for (int i = _g->nxs(j); i < _g->nxe(j); i++) {
        int l2d = _g->ijtol(i, j);
        int l   = l2d + k * _g->nxy();

        invmat2g(&diag(0, 0, l), &delinv(0, 0, l));

        auto le = _g->neib(EAST, l);

        if (le == -1) continue;

        for (int ig = 0; ig < _g->ng(); ++ig) {
            delcc(0, ig, le) = cc(LEFT, XDIR, ig, le) * delinv(0, ig, l);
            delcc(1, ig, le) = cc(LEFT, XDIR, ig, le) * delinv(1, ig, l);
        }

        // diag(0, 0, le) = diag(0, 0, le) - delcc(0, 0, le) * cc(RIGHT, XDIR, 0, l);
        // diag(1, 0, le) = diag(1, 0, le) - delcc(1, 0, le) * cc(RIGHT, XDIR, 1, l);
        // diag(0, 1, le) = diag(0, 1, le) - delcc(0, 1, le) * cc(RIGHT, XDIR, 0, l);
        // diag(1, 1, le) = diag(1, 1, le) - delcc(1, 1, le) * cc(RIGHT, XDIR, 1, l);
    }
}

void LineJacobiBicgSolver::solve(double* diag, double* cc, double& r20, double* flux, double& r2) {
    int n = _g->nxyz() * _g->ng();

    // solves the linear system by preconditioned BiCGSTAB Algorithm
    double crhod = _crho;
    _crho        = myblas::dot(n, _vr0, _vr);

    _cbeta = _crho * _calpha / (crhod * _comega);

    //    _vp(:,:,:)=_vr(:,:,:)+_cbeta*(_vp(:,:,:)-_comega*_vv(:,:,:))

    myblas::multi(n, _comega, _vv, _vt);
    myblas::minus(n, _vp, _vt, _vt);
    myblas::multi(n, _cbeta, _vt, _vt);
    myblas::plus(n, _vr, _vt, _vp);

    minv(cc, _vp, _vy);
    axb(diag, cc, _vy, _vv);

    double r0v = myblas::dot(n, _vr0, _vv);

    if (r0v == 0.0) {
        return;
    }

    _calpha = _crho / r0v;

    //    _vs(:,:,:)=_vr(:,:,:)-_calpha*_vv(:,:,:)
    myblas::multi(n, _calpha, _vv, _vt);
    myblas::minus(n, _vr, _vt, _vs);

    minv(cc, _vs, _vz);
    axb(diag, cc, _vz, _vt);

    double pts = myblas::dot(n, _vs, _vt);
    double ptt = myblas::dot(n, _vt, _vt);

    _comega = 0.0;
    if (ptt != 0.0) {
        _comega = pts / ptt;
    }

    //    flux(:, :, :) = flux(:, :, :) + _calpha * _vy(:,:,:)+_comega * _vz(:,:,:)
    myblas::multi(n, _comega, _vz, _vz);
    myblas::multi(n, _calpha, _vy, _vy);
    myblas::plus(n, _vz, _vy, _vy);
    myblas::plus(n, flux, _vy, flux);

    //    _vr(:,:,:)=_vs(:,:,:)-_comega * _vt(:,:,:)
    myblas::multi(n, _comega, _vt, _vr);
    myblas::minus(n, _vs, _vr, _vr);

    if (r20 != 0.0) {
        r2 = sqrt(myblas::dot(n, _vr, _vr)) / r20;
    }
}

void LineJacobiBicgSolver::axb(double* diag, double* cc, double* flux, double* aflux) {

#pragma omp parallel for
    for (int l = 0; l < _g->nxyz(); ++l) {
        for (int ig = 0; ig < _g->ng(); ++ig) {
            aflux(ig, l) = axb(ig, l, diag, cc, flux);
        }
    }
}

double LineJacobiBicgSolver::axb(const int& ig, const int& l, double* diag, double* cc, double* flux) {

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
