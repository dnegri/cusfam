//
// Created by JOO IL YOON on 2021/01/30.
//

#include "JacobiBicgSolver.h"
#include "mat2g.h"
#include "myblas.h"

#define diag(igs, ige, l) diag[l*_g->ng2()+ige*_g->ng()+igs]
#define cc(lr, idir, ig, l) cc[l*_g->ng()*NDIRMAX*LR+ig*NDIRMAX*LR+idir*LR+lr]
#define src(ig, l) src[l*_g->ng()+ig]
#define aflux(ig, l) aflux[l*_g->ng()+ig]
#define b(ig, l) b[l*_g->ng()+ig]
#define x(ig, l) x[l*_g->ng()+ig]
#define flux(ig, l) flux[l*_g->ng()+ig]
#define b1d(ig, l)   b1d[(l*_g->ng())+ig]
#define x1d(ig, l)   x1d[(l*_g->ng())+ig]
#define b01d(ig, l)  _b01d[(l*_g->ng())+ig]
#define s1dl(ig, l)  _s1dl[(l*_g->ng())+ig]
#define b03d(ig, l)  _b03d[(l*_g->ng())+ig]
#define s3d(ig, l)  _s3d[(l*_g->ng())+ig]
#define s3dd(ig, l)  _s3dd[(l*_g->ng())+ig]


#define vr(ig, l)   _vr[(l*_g->ng())+ig]
#define vr0(ig, l)  _vr0[(l*_g->ng())+ig]
#define vp(ig, l)   _vp[(l*_g->ng())+ig]
#define vv(ig, l)   _vv[(l*_g->ng())+ig]
#define vs(ig, l)   _vs[(l*_g->ng())+ig]
#define vt(ig, l)   _vt[(l*_g->ng())+ig]
#define vy(ig, l)   _vy[(l*_g->ng())+ig]
#define vz(ig, l)   _vz[(l*_g->ng())+ig]
#define y1d(ig, l)   _y1d[(l*_g->ng())+ig]
#define b1i(ig, l)   _b1i[(l*_g->ng())+ig]

#define del(igs, ige, l)  _del[(l*_g->ng2())+(ige)*_g->ng()+(igs)]
#define ainvd(igs, ige, l)    _ainvd[(l*_g->ng2())+(ige)*_g->ng()+(igs)]
#define ainvl(igs, ige, l)    _ainvl[(l*_g->ng2())+(ige)*_g->ng()+(igs)]
#define ainvu(igs, ige, l)    _ainvu[(l*_g->ng2())+(ige)*_g->ng()+(igs)]
#define au(igs, ige, l)   _au[(l*_g->ng2())+(ige)*_g->ng()+(igs)]
#define delinv(igs, ige, l)   _delinv[(l*_g->ng2())+(ige)*_g->ng()+(igs)]
#define al(igs, ige, l)       _al[(l*_g->ng2())+(ige)*_g->ng()+(igs)]
#define deliau(igs, ige, l)   _deliau[(l*_g->ng2())+(ige)*_g->ng()+(igs)]


JacobiBicgSolver::JacobiBicgSolver(Geometry& g) {

    _g = &g;

    _calpha = 0.0;
    _cbeta = 0.0;
    _crho = 0.0;
    _comega = 0.0;


    _vz = new SOL_VAR[_g->ng() * _g->nxyz()]{};
    _vy = new SOL_VAR[_g->ng() * _g->nxyz()]{};

    _vr = new CMFD_VAR[_g->ng() * _g->nxyz()]{};
    _vr0 = new CMFD_VAR[_g->ng() * _g->nxyz()]{};
    _vp = new CMFD_VAR[_g->ng() * _g->nxyz()]{};
    _vv = new CMFD_VAR[_g->ng() * _g->nxyz()]{};
    _vs = new CMFD_VAR[_g->ng() * _g->nxyz()]{};
    _vt = new CMFD_VAR[_g->ng() * _g->nxyz()]{};
    _delinv = new CMFD_VAR[_g->ng2() * _g->nxyz()]{};
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
}

float JacobiBicgSolver::reset(const int& l, CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux, CMFD_VAR* src) {

    float r = 0.0;
    for (int ig = 0; ig < _g->ng(); ig++)
    {
        float aflux = axb(ig, l, diag, cc, flux);
        vr(ig, l) = src(ig, l) - aflux;
        vr0(ig, l) = vr(ig, l);
        vp(ig, l) = 0.0;
        vv(ig, l) = 0.0;
        r += vr(ig, l) * vr(ig, l);
    }

    return r;
}

void JacobiBicgSolver::reset(CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux, CMFD_VAR* src, float& r20) {

    _calpha = 1;
    _crho = 1;
    _comega = 1;

    r20 = 0;
    for (int l = 0; l < _g->nxyz(); ++l) {
        r20 += reset(l, diag, cc, flux, src);
    }

    r20 = sqrt(r20);
}

void JacobiBicgSolver::minv(CMFD_VAR* cc, CMFD_VAR* b, SOL_VAR* x) {

#pragma omp parallel for
    for (int k = 0; k < _g->nz(); ++k) {
        for (int l2d = 0; l2d < _g->nxy(); ++l2d) {
            int l = k*_g->nxy()+l2d;
            minv(l, cc, b, x);
        }
    }
}

void JacobiBicgSolver::minv(const int & l , CMFD_VAR* cc, CMFD_VAR* b, SOL_VAR* x) {

    x(0, l) = delinv(0, 0, l) * b(0, l) + delinv(1, 0, l) * b(1, l);
    x(1, l) = delinv(0, 1, l) * b(0, l) + delinv(1, 1, l) * b(1, l);
}

void JacobiBicgSolver::facilu(CMFD_VAR* diag, CMFD_VAR* cc) {

#pragma omp parallel for
    for (int k = 0; k < _g->nz(); ++k) {
        for (int l2d = 0; l2d < _g->nxy(); ++l2d) {
            int l = k*_g->nxy()+l2d;
            facilu(l, diag, cc);
        }
    }
}
void JacobiBicgSolver::facilu(const int& l, CMFD_VAR* diag, CMFD_VAR* cc) {

    invmat2g(&diag(0, 0, l), &delinv(0, 0, l));
}

void JacobiBicgSolver::solve(CMFD_VAR* diag, CMFD_VAR* cc, float& r20, SOL_VAR* flux, float& r2) {
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

    CMFD_VAR r0v = myblas::dot(n, _vr0, _vv);

    if (r0v == 0.0) {
        return;
    }

    _calpha = _crho / r0v;

//    _vs(:,:,:)=_vr(:,:,:)-_calpha*_vv(:,:,:)
    myblas::multi(n, _calpha, _vv, _vt);
    myblas::minus(n, _vr, _vt, _vs);

    minv(cc, _vs, _vz);
    axb(diag, cc, _vz, _vt);

    CMFD_VAR pts = myblas::dot(n, _vs, _vt);
    CMFD_VAR ptt = myblas::dot(n, _vt, _vt);

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
        r2 = sqrt(myblas::dot(n, _vt, _vt)) / r20;
    }
}

void JacobiBicgSolver::axb(CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux, CMFD_VAR* aflux) {
    for (int l = 0; l < _g->nxyz(); ++l) {
        for (int ig = 0; ig < _g->ng(); ++ig) {
            aflux(ig, l) = axb(ig, l, diag, cc, flux);
        }
    }
}

CMFD_VAR JacobiBicgSolver::axb(const int& ig, const int& l, CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux) {

    CMFD_VAR ab = 0.0;

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
