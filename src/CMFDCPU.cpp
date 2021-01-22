//
// Created by JOO IL YOON on 2021/01/21.
//

#include "CMFDCPU.h"

#define flux(ig, l)   (flux[(l)*_g.ng()+ig])
#define aflux(ig, l)   (aflux[(l)*_g.ng()+ig])
#define psi(l)   (_psi[(l)])

CMFDCPU::CMFDCPU(Geometry &g, CrossSection &x) : CMFD(g, x) {
}
CMFDCPU::~CMFDCPU() {

}

void CMFDCPU::upddtil() {
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddtil(ls);
    }
}
void CMFDCPU::upddhat() {
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddtil(ls);
    }

}

void CMFDCPU::setls() {
    for (int l = 0; l < _g.nxyz(); ++l) {
        CMFD::setls(l);
    }
}

void CMFDCPU::axb(double * flux, double* aflux) {
    for (int l = 0; l < _g.nxyz(); ++l) {
        for (int ig = 0; ig < _g.ng(); ++ig) {
            aflux(ig, l) = am(0,ig,l) * flux(0, l) + am(1, ig, l) * flux(1, l);

            for (int idir = 0; idir < NDIRMAX; ++idir) {
                for (int lr = 0; lr < LR; ++lr) {
                    int ln = _g.neib(lr,idir,l);
                    if(ln != -1)
                        aflux(ig, l) += cc(lr,idir,ig,l) * flux(ig,ln);
                }
            }
        }
    }
}

double CMFDCPU::wiel(const int& icy, double* flux, double* psi, double& eigv, double& reigv)
{
    double errl2 = 0;

    double gamman = 0;
    double gammad = 0;

    for (size_t l = 0; l < _g.nxyz(); l++)
    {
        double psid = psi(l);
        psi(l) = _x.xsnf(1, l) * flux(1, l) + _x.xsnf(2, l) * flux(2, l);
        psi(l) = psi(l) * _g.vol(l);

        double err = psi(l, k) - psid;
        errl2 = errl2 + err * err;
        gammad += psid * psi(l);
        gamman += psi(l) * psi(l);
    }

    //compute new eigenvalue
    double eigvd = eigv;
    if (icy < 0) {
        double sumf = 0;
        double summ = 0;
        for (size_t l = 0; l < _g.nxyz(); l++)
        {
            for (size_t ig = 0; ig < _g.ng(); ig++)
            {
                double ab = CMFD::axb(ig, l, flux);
                summ = summ + ab;
            }
            sumf += psi(l);
            summ += psi(l) * _reigvs;
        }
    }
    else {
        double gamma = gammad / gamman;
        eigv = 1 / (reigv * gamma + (1 - gamma) * _reigvs);
        reigv = 1 / eigv;
    }

    errl2 = sqrt(errl2 / gammad);
    double erreig = abs(eigv - eigvd);;

    _eigshft = 0;
    if (icy >= 0) {
        _eigshft = _eshift;
    }
    _eigvs = eigv + _eigshft;
    _reigvsd = _reigvs;
    _reigvs = 0;
    if (_eigshft != 0.0) _reigvs = 1 / _eigvs;
    
    return errl2;
}


double CMFDCPU::residual(const double& reigv, const double& reigvs, double* flux) {

    double reigvdel=reigv-reigvs;

//    axb(phi,aphi);
    double r = 0.0;
    double psi2 = 0.0;

    for (int l = 0; l < _g.nxyz(); ++l) {
        double fs=psi(l)*reigvdel;

        for (int ig = 0; ig < _g.ng(); ++ig) {
            double ab = CMFD::axb(ig, l, flux);

            double err = _x.chif(ig,l)*fs-ab;
            r += err*err;

            double ps = _x.chif(ig,l)*psi(l);
            psi2 += ps*ps;
        }
    }

    return sqrt(r/psi2);
}

void CMFDCPU::drive(double& eigv, double* flux, float& errl2) {

    int icy     = 0;
    int icmfd   = 0;
    double reigv = 1./eigv;
    double reigvsdel=0, reigvsd=0;
    double resid0;

    for (int iout = 0; iout < _ncmfd; ++iout) {
        icy=icy+1;
        icmfd=icmfd+1;
        double eigvd = eigv;
        double reigvdel=reigv -_reigvs;
        for (int l = 0; l < _g.nxyz(); ++l) {
            double fs=psi(l)*reigvdel;
            for (int ig = 0; ig < _g.ng(); ++ig) {
                src(ig,l)=_x.chif(ig,l)*fs;
            }
        }

        //solve linear system A*phi = src
        // update flux
        //_ls->solve(a, src, flux);

        //wielandt shift
        //wiel(icy, phi, psi, eigv, reigv, errl2, errlinf);

        reigvsdel=_reigvs - reigvsd;
        reigvdel=reigv-_reigvs;

        for (int l = 0; l < _g.nxyz(); ++l) {
            am(0,0,l)=am(0,0,l)-_x.xsnf(0,l)*_g.vol(l)*reigvsdel*_x.chif(0,l);
            am(1,1,l)=am(1,1,l)-_x.xsnf(1,l)*_g.vol(l)*reigvsdel*_x.chif(1,l);
            am(1,0,l) = -_x.xssf(1,0,l)*_g.vol(l) - af(1,l)*_reigvs;
            am(0,1,l) = -_x.xssf(0,1,l)*_g.vol(l) - af(0,l)*_reigvs;

        }
        reigvsd=_reigvs;

        double resi = residual(reigv,_reigvs, flux);

        if(icmfd == 0) resid0 = resi;
        double relresid = resi/resid0;

        int negative=0;
        for (int l = 0; l < _g.nxyz(); ++l) {
            for (int ig = 0; ig < _g.ng(); ++ig) {
                if(flux(ig,l) < 0) {
                    ++negative;
                }
            }
        }

        double erreig = abs(eigv - eigvd);

        if (errl2 < _epsl2) break;

    }
}


