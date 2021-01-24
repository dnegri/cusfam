//
// Created by JOO IL YOON on 2021/01/21.
//

#include "CMFDCPU.h"

#define flux(ig, l)   (flux[(l)*_g.ng()+ig])
#define aflux(ig, l)   (aflux[(l)*_g.ng()+ig])
#define psi(l)  (psi[(l)])


CMFDCPU::CMFDCPU(Geometry &g, CrossSection &x) : CMFD(g, x) {

}

CMFDCPU::~CMFDCPU() {

}

void CMFDCPU::upddtil() {
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddtil(ls);
    }
}

void CMFDCPU::upddhat(double* flux, float* jnet) {
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddhat(ls, flux, jnet);
    }

}

void CMFDCPU::setls() {
    for (int l = 0; l < _g.nxyz(); ++l) {
        setls(l);
    }
}

void CMFDCPU::setls(const int &l) {
    CMFD::setls(l);

    int idxrow = l * _g.ng();

    for (int ige = 0; ige < _g.ng(); ++ige) {
        int irow = _ls->rowptr(idxrow + ige);

        for (int idir = NDIRMAX - 1; idir >= 0; --idir) {
            int ln = _g.neib(LEFT, idir, l);

            if (ln >= 0) {
                _ls->a(irow++) = cc(LEFT, idir, ige, l);
            }
        }

        for (int igs = 0; igs < _g.ng(); igs++) {
            _ls->a(irow++) = diag(igs, ige, l);
        }

        for (int idir = 0; idir < NDIRMAX; idir++) {
            int ln = _g.neib(RIGHT, idir, l);

            if (ln >= 0) {
                _ls->a(irow++) = cc(RIGHT, idir, ige, l);
            }
        }
    }

}

void CMFDCPU::updls(const double &reigvs) {
    for (int l = 0; l < _g.nxyz(); ++l) {
        updls(l, reigvs);
    }
}

void CMFDCPU::updls(const int &l, const double &reigvs) {
    int idxrow = l * _g.ng();

    for (int ige = 0; ige < _g.ng(); ++ige) {
        int idx = _ls->idxdiag(idxrow + ige);
        for (int igs = 0; igs < _g.ng(); ++igs) {
            _ls->a(idx + igs) = diag(igs, ige, l) - (_x.chif(ige, l) * _x.xsnf(igs, l) * reigvs * _g.vol(l));
        }
    }
}

void CMFDCPU::axb(double *flux, double *aflux) {
    for (int l = 0; l < _g.nxyz(); ++l) {
        for (int ig = 0; ig < _g.ng(); ++ig) {
            aflux(ig, l) = CMFD::axb(ig, l, flux);
        }
    }
}

double CMFDCPU::wiel(const int &icy, double *flux, double *psi, double &eigv, double &reigv, double &reigvs) {
    double errl2 = 0;

    double gamman = 0;
    double gammad = 0;

    for (size_t l = 0; l < _g.nxyz(); l++) {
        double psid = psi(l);
        psi(l) = _x.xsnf(1, l) * flux(1, l) + _x.xsnf(2, l) * flux(2, l);
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
    } else {
        double gamma = gammad / gamman;
        eigv = 1 / (reigv * gamma + (1 - gamma) * reigvs);
        reigv = 1 / eigv;
    }

    errl2 = sqrt(errl2 / gammad);
    double erreig = abs(eigv - eigvd);;

    double eigvs = eigv;
    if (icy >= 0) {
        eigvs += _eshift;
    }

    reigvs = 0;
    if (_eshift != 0.0) reigvs = 1 / eigvs;

    return errl2;
}


double CMFDCPU::residual(const double &reigv, const double &reigvs, double *flux, double *psi) {

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

void CMFDCPU::drive(double &eigv, double *flux, double *psi, float &errl2) {

    int icy = 0;
    int icmfd = 0;
    double reigv = 1. / eigv;
    double reigvs = 0.0;
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

        //solve linear system A*phi = src
        // update flux
        _ls->solve(_src, flux);

        //wielandt shift
        errl2 = wiel(icy, flux, psi, eigv, reigv, reigvs);

        updls(reigvs);

        double resi = residual(reigv, reigvs, flux, psi);

        if (icmfd == 0) resid0 = resi;
        double relresid = resi / resid0;

        int negative = 0;
        for (int l = 0; l < _g.nxyz(); ++l) {
            for (int ig = 0; ig < _g.ng(); ++ig) {
                if (flux(ig, l) < 0) {
                    ++negative;
                }
            }
        }

        printf("EIGV : %9.5f , ERRL2 : %12.5E\n", eigv, errl2);

        if (errl2 < _epsl2) break;

    }
}




