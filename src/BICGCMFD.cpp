//
// Created by JOO IL YOON on 2021/01/31.
//

#include "BICGCMFD.h"

#define flux(ig, l)   (flux[(l)*_g.ng()+ig])
#define aflux(ig, l)   (aflux[(l)*_g.ng()+ig])

BICGCMFD::BICGCMFD(Geometry &g, CrossSection &x) : CMFD(g, x) {
}

void BICGCMFD::init()
{
    CMFD::init();
    _ls = new JacobiBicgSolver(_g);
    _nodal = new NodalCPU(_g, _x);
    _nodal->init();

    _epsbicg = 1.E-4;
    _nmaxbicg = 5;
    _eshift = 0.01;
    iter = 0;

    _unshifted_diag = new CMFD_VAR[_g.ng2() * _g.nxyz()];
}

BICGCMFD::~BICGCMFD() {
    delete _ls;
    delete[] _unshifted_diag;
}

void BICGCMFD::setEshift(float eshift) {
    _eshift = eshift;
}

void BICGCMFD::wiel(const int& icy, const SOL_VAR* flux, double& reigvs, double& eigv, double& reigv, float& errl2) {

    double gamman = 0;
    double gammad = 0;
    float  err = 0;

#pragma omp parallel for reduction(+ : gammad, gamman, err)
    for (int l = 0; l < _g.nxyz(); l++) {
        CMFD_VAR psid = psi(l);
        psi(l) = _x.xsnf(0, l) * flux(0, l) + _x.xsnf(1, l) * flux(1, l);
        psi(l) = psi(l) * _g.vol(l);

        float err1 = psi(l) - psid;
        err = err + err1 * err1;
        gammad += psid * psi(l);
        gamman += psi(l) * psi(l);
    }

    errl2 = err;

    //compute new eigenvalue
    double eigvd = eigv;
    if (icy < 0) {
        double sumf = 0;
        double summ = 0;
        #pragma omp parallel for reduction(+ : summ, sumf)
        for (int l = 0; l < _g.nxyz(); l++) {
            for (int ig = 0; ig < _g.ng(); ig++) {
                CMFD_VAR ab = CMFD::axb(ig, l, flux);
                summ = summ + ab;
            }
            sumf += psi(l);
            summ += psi(l) * reigvs;
        }
        eigv = sumf / summ;
    }
    else {
        double gamma = gammad / gamman;
        eigv = 1 / (reigv * gamma + (1 - gamma) * reigvs);
    }
    reigv = 1 / eigv;

    errl2 = sqrt(errl2 / gammad);
    double erreig = abs(eigv - eigvd);;

    double eigvs = eigv;
    if (icy >= 0) {
        eigvs += _eshift;
    }

    reigvs = 0;
    if (_eshift != 0.0) reigvs = 1 / eigvs;

}


double BICGCMFD::residual(const double& reigv, const double& reigvs, const SOL_VAR* flux) {

    double reigvdel = reigv - reigvs;

//    axb(phi,aphi);
    double r = 0.0;
    double psi2 = 0.0;

#pragma omp parallel for reduction(+ : r, psi2)
    for (int l = 0; l < _g.nxyz(); ++l) {
        double fs = psi(l) * reigvdel;

        for (int ig = 0; ig < _g.ng(); ++ig) {
            double ab = CMFD::axb(ig, l, flux);

            double err = _x.chif(ig, l) * fs - ab;
            r += err * err;

            double ps = _x.chif(ig, l) * psi(l);
            psi2 += ps * ps;
        }
    }

    return sqrt(r / psi2);
}

void BICGCMFD::upddtil() {
    #pragma omp parallel for 
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddtil(ls);
    }
}

void BICGCMFD::upddhat(SOL_VAR* flux, SOL_VAR* jnet) {
#pragma omp parallel for 
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddhat(ls, flux, jnet);
    }

}

void BICGCMFD::setls(const double& eigv) {
    double reigvs = 0.0;
    if(_eshift != 0.0) reigvs = 1. / (eigv + _eshift);

    for (int l = 0; l < _g.nxyz(); ++l) {
        setls(l);
        updls(l, reigvs);
    }
    _ls->facilu(_diag, _cc);
}

void BICGCMFD::setls(const int &l) {
    CMFD::setls(l);
    for (int ige = 0; ige < _g.ng(); ++ige) {
        for (int igs = 0; igs < _g.ng(); ++igs) {
            unshifted_diag(igs, ige, l) = diag(igs, ige, l);
        }
    }
}

void BICGCMFD::updls(const double& reigvs) {
    for (int l = 0; l < _g.nxyz(); ++l) {
        updls(l, reigvs);
    }
}
void BICGCMFD::updls(const int &l, const double& reigvs) {
    for (int ige = 0; ige < _g.ng(); ++ige) {
        for (int igs = 0; igs < _g.ng(); ++igs) {
            diag(igs,ige,l) = unshifted_diag(igs, ige, l) - (_x.chif(ige, l) * _x.xsnf(igs, l) * reigvs * _g.vol(l));
        }
    }
}

void BICGCMFD::updjnet(SOL_VAR* flux, SOL_VAR* jnet)
{
#pragma omp parallel for 
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::updjnet(ls, flux, jnet);
    }
}

void BICGCMFD::updpsi(const SOL_VAR* flux)
{
#pragma omp parallel for 
    for (int l = 0; l < _g.nxyz(); ++l) {
        CMFD::updpsi(l, flux);
    }
}


void BICGCMFD::axb(SOL_VAR* flux, SOL_VAR* aflux) {
    #pragma omp parallel for 
    for (int l = 0; l < _g.nxyz(); ++l) {
        for (int ig = 0; ig < _g.ng(); ++ig) {
            aflux(ig, l) = CMFD::axb(ig, l, flux);
        }
    }
}


void BICGCMFD::drive(double &eigv, SOL_VAR* flux, float &errl2) {

    int icmfd = 0;
    double reigv = 1. / eigv;
    double reigvs = 0.0;

    if(_eshift != 0.0) reigvs = 1. / (eigv + _eshift);

    for (int iout = 0; iout < _ncmfd; ++iout) {
        ++iter; ++icmfd;
        double reigvdel = reigv - reigvs;

        #pragma omp parallel for 
        for (int l = 0; l < _g.nxyz(); ++l) {
            double fs = psi(l) * reigvdel;
            for (int ig = 0; ig < _g.ng(); ++ig) {
                src(ig, l) = _x.chif(ig, l) * fs;
            }
        }

        CMFD_VAR r20=0.0;
        _ls->reset(_diag, _cc, flux, _src, r20);

        CMFD_VAR r2 = 0.0;
        for (int iin = 0; iin < _nmaxbicg; ++iin) {
            //solve linear system A*phi = src
            _ls->solve(_diag, _cc, r20, flux, r2);
            //printf("JacobiBicgSolver Iteration : %d   Error : %e\n", iin, r2);
            if(r2 < _epsbicg) break;
        }

        //wielandt shift
        wiel(iter, flux, reigvs, eigv, reigv, errl2);

        if(reigvs != 0.0) updls(reigvs);

        int negative = 0;
#pragma omp parallel for reduction(+ : negative)
        for (int l = 0; l < _g.nxyz(); ++l) {
            for (int ig = 0; ig < _g.ng(); ++ig) {
                if (flux(ig, l) < 0) {
                    ++negative;
                }
            }
        }
        if(negative == _g.ngxyz()) {
            negative = 0;
        }

        if(negative != 0 && icmfd < 5*_ncmfd) iout--;

        printf("IOUT : %d, EIGV : %9.7f , ERRL2 : %12.5E, NEGATIVE : %d\n", iter, eigv, errl2, negative);

        if (errl2 < _epsl2) break;

    }
}

void BICGCMFD::resetIteration() {
    iter = 0;
}

void BICGCMFD::updnodal(double& eigv, SOL_VAR* flux, SOL_VAR* jnet) {
    updjnet(flux, jnet);
    _nodal->reset(_x, 1./eigv, jnet, flux);
    _nodal->drive(jnet);
    upddhat(flux, jnet);
}


