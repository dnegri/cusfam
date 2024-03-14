//
// Created by JOO IL YOON on 2021/01/30.
//
#include "JacobiBicgSolver.h"
#include "mat2g.h"
#include "myblas.h"

#define diag(igs, ige, l)   diag[l*_g->ng2()+ige*_g->ng()+igs]
#define cc(lr, idir, ig, l) cc[l*_g->ng()*NDIRMAX*LR+ig*NDIRMAX*LR+idir*LR+lr]
#define src(ig, l)      src[l*_g->ng()+ig]
#define aflux(ig, l)    aflux[l*_g->ng()+ig]
#define b(ig, l)        b[l*_g->ng()+ig]
#define x(ig, l)        x[l*_g->ng()+ig]
#define flux(ig, l) flux[l*_g->ng()+ig]


JacobiBicgSolver::JacobiBicgSolver(Geometry& g) {

    _g = &g;

    _calpha = 0.0;
    _cbeta = 0.0;
    _crho = 0.0;
    _comega = 0.0;


    _vz = new double[_g->ng() * _g->nxyz()]{};
    _vy = new double[_g->ng() * _g->nxyz()]{};

    _vr = new double[_g->ng() * _g->nxyz()]{};
    _vr0 = new double[_g->ng() * _g->nxyz()]{};
    _vp = new double[_g->ng() * _g->nxyz()]{};
    _vv = new double[_g->ng() * _g->nxyz()]{};
    _vs = new double[_g->ng() * _g->nxyz()]{};
    _vt = new double[_g->ng() * _g->nxyz()]{};
    _delinv = new double[_g->ng2() * _g->nxyz()]{};
}

JacobiBicgSolver::~JacobiBicgSolver() {
    delete _vr;
    delete _vr0;
    delete _vp;
    delete _vv;
    delete _vs;
    delete _vt;
    delete _vy;
    delete _vz;
    delete _delinv;
}

double JacobiBicgSolver::reset(const int& l, double* diag, double* cc, double* flux, double* src) {

    double r = 0.0;
    for (int ig = 0; ig < _g->ng(); ig++)
    {
        double aflux = axb(ig, l, diag, cc, flux);
        vr(ig, l) = src(ig, l) - aflux;
        vr0(ig, l) = vr(ig, l);
        vp(ig, l) = 0.0;
        vv(ig, l) = 0.0;
        r += vr(ig, l) * vr(ig, l);
    }

    return r;
}

void JacobiBicgSolver::reset(double* diag, double* cc, double* flux, double* src, double& r20) {

    _calpha = 1;
    _crho = 1;
    _comega = 1;

    r20 = 0;

	#pragma omp parallel for reduction (+ : r20)
    for (int l = 0; l < _g->nxyz(); ++l) {
        r20 += reset(l, diag, cc, flux, src);
    }

    r20 = sqrt(r20);
}

void JacobiBicgSolver::minv(double* cc, double* b, double* x) {

#pragma omp parallel for
    for (int k = 0; k < _g->nz(); ++k) {
        for (int l2d = 0; l2d < _g->nxy(); ++l2d) {
            int l = k*_g->nxy()+l2d;
            minv(l, cc, b, x);
        }
    }
}

void JacobiBicgSolver::minv(const int & l , double* cc, double* b, double* x) {

    x(0, l) = delinv(0, 0, l) * b(0, l) + delinv(1, 0, l) * b(1, l);
    x(1, l) = delinv(0, 1, l) * b(0, l) + delinv(1, 1, l) * b(1, l);
}

void JacobiBicgSolver::facilu(double* diag, double* cc) {

#pragma omp parallel for
    for (int k = 0; k < _g->nz(); ++k) {
        for (int l2d = 0; l2d < _g->nxy(); ++l2d) {
            int l = k*_g->nxy()+l2d;
            facilu(l, diag, cc);
        }
    }
}
void JacobiBicgSolver::facilu(const int& l, double* diag, double* cc) {

    invmat2g(&diag(0, 0, l), &delinv(0, 0, l));
}

void JacobiBicgSolver::solve(double* diag, double* cc, double& r20, double* flux, double& r2) {
    int n = _g->nxyz() * _g->ng();

    // solves the linear system by preconditioned BiCGSTAB Algorithm
	double crhod = _crho;
    _crho = myblas::dot(n, _vr0, _vr);

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

void JacobiBicgSolver::axb(double* diag, double* cc, double* flux, double* aflux) {
	#pragma omp parallel for
    for (int l = 0; l < _g->nxyz(); ++l) {
        for (int ig = 0; ig < _g->ng(); ++ig) {
            aflux(ig, l) = axb(ig, l, diag, cc, flux);
        }
    }
}

double JacobiBicgSolver::axb(const int& ig, const int& l, double* diag, double* cc, double* flux) {

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
