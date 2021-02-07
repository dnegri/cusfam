//
// Created by JOO IL YOON on 2021/01/31.
//

#include "BICGCMFD.h"

#define flux(ig, l)   (flux[(l)*_g.ng()+ig])
#define aflux(ig, l)   (aflux[(l)*_g.ng()+ig])

BICGCMFD::BICGCMFD(Geometry &g, CrossSection &x) : CMFD(g, x) {
    _ls = new BICGSolver(g);
    _epsbicg = 1.E-4;
    _nmaxbicg = 3;

    _eshift_diag = new double[g.ng2() * g.nxyz()];
    _eshift = 0.0;

}

BICGCMFD::~BICGCMFD() {
    delete _ls;
    delete[] _eshift_diag;
}

void BICGCMFD::setEshift(float eshift) {
    _eshift = eshift;
}

void BICGCMFD::wiel(const int& icy, const double* flux, double& reigvs, double& eigv, double& reigv, float& errl2) {

    double gamman = 0;
    double gammad = 0;
    errl2 = 0;

    for (size_t l = 0; l < _g.nxyz(); l++) {
        double psid = psi(l);
        psi(l) = _x.xsnf(0, l) * flux(0, l) + _x.xsnf(1, l) * flux(1, l);
        psi(l) = psi(l) * _g.vol(l);

        double err = psi(l) - psid;
        errl2 = errl2 + err * err;
        gammad += psid * psi(l);
        gamman += psi(l) * psi(l);
    }

    //compute new eigenvalue
    double eigvd = eigv;
    if (icy < 0) {
        double sumf = 0;
        double summ = 0;
        for (size_t l = 0; l < _g.nxyz(); l++) {
            for (size_t ig = 0; ig < _g.ng(); ig++) {
                double ab = CMFD::axb(ig, l, flux);
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


double BICGCMFD::residual(const double& reigv, const double& reigvs, const double* flux) {

    double reigvdel = reigv - reigvs;

//    axb(phi,aphi);
    double r = 0.0;
    double psi2 = 0.0;

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
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddtil(ls);
    }
}

void BICGCMFD::upddhat(double* flux, double* jnet) {
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
            eshift_diag(igs, ige, l) = diag(igs, ige, l);
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
            diag(igs,ige,l) = eshift_diag(igs, ige, l) - (_x.chif(ige, l) * _x.xsnf(igs, l) * reigvs * _g.vol(l));
        }
    }
}

void BICGCMFD::updjnet(double* flux, double* jnet)
{
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::updjnet(ls, flux, jnet);
    }
}

void BICGCMFD::updpsi(const double* flux)
{
    for (int l = 0; l < _g.nxyz(); ++l) {
        CMFD::updpsi(l, flux);
    }
}


void BICGCMFD::axb(double* flux, double* aflux) {
    for (int l = 0; l < _g.nxyz(); ++l) {
        for (int ig = 0; ig < _g.ng(); ++ig) {
            aflux(ig, l) = CMFD::axb(ig, l, flux);
        }
    }
}


void BICGCMFD::drive(double &eigv, double *flux, float &errl2) {

    int icy = 0;
    int icmfd = 0;
    double reigv = 1. / eigv;
    double reigvs = 0.0;

    if(_eshift != 0.0) reigvs = 1. / (eigv + _eshift);
    double resid0;

    for (int iout = 0; iout < _ncmfd; ++iout) {
        icy = icy + 1;
        icmfd = icmfd + 1;
        double reigvdel = reigv - reigvs;
        for (int l = 0; l < _g.nxyz(); ++l) {
            double fs = psi(l) * reigvdel;
            for (int ig = 0; ig < _g.ng(); ++ig) {
                src(ig, l) = _x.chif(ig, l) * fs;
            }
        }

        double r20=0.0;
        _ls->reset(_diag, _cc, flux, _src, r20);

        double r2 = 0.0;
        for (int iin = 0; iin < _nmaxbicg; ++iin) {
            //solve linear system A*phi = src
            _ls->solve(_diag, _cc, r20, flux, r2);
            if(r2 < _epsbicg) break;
        }

        //wielandt shift
        wiel(icy, flux, reigvs, eigv, reigv, errl2);

        if(reigvs != 0.0) updls(reigvs);

        int negative = 0;
        for (int l = 0; l < _g.nxyz(); ++l) {
            for (int ig = 0; ig < _g.ng(); ++ig) {
                if (flux(ig, l) < 0) {
                    ++negative;
                }
            }
        }

        printf("IOUT : %d, EIGV : %9.7f , ERRL2 : %12.5E, NEGATIVE : %d\n", iout, eigv, errl2, negative);

        if (errl2 < _epsl2) break;

    }
}


