#include "CMFD.h"


#define flux(ig, l)   (flux[(l)*_g.ng()+ig])
#define jnet(ig, ls)      (jnet[(ls)*_g.ng() + ig])

CMFD::CMFD(Geometry &g, CrossSection& x) : _g(g), _x(x){

}

CMFD::~CMFD() {
    delete[] _dtil;
    delete[] _dhat;
    delete[] _diag;
    delete[] _cc;
    delete[] _src;
    delete[] _psi;
}

void CMFD::init()
{
    _epsl2 = 1.E-5;
    _dtil = new CMFD_VAR[_g.nsurf() * _g.ng()]{};
    _dhat = new CMFD_VAR[_g.nsurf() * _g.ng()]{};
    _diag = new CMFD_VAR[_g.nxyz() * _g.ng2()]{};
    _cc = new CMFD_VAR[_g.nxyz() * _g.ng() * NEWSBT]{};
    _src = new CMFD_VAR[_g.nxyz() * _g.ng()]{};
    _psi = new CMFD_VAR[_g.nxyz()]{};
}

void CMFD::upddtil(const int& ls)
{
    int ll = _g.lklr(LEFT,ls);
    int lr = _g.lklr(RIGHT,ls);
    int idirl = _g.idirlr(LEFT,ls);
    int idirr = _g.idirlr(RIGHT,ls);

	float betal, betar;

	for (int ig = 0; ig < _g.ng(); ig++)
	{
	    if(ll < 0) {
            betal = _g.albedo(LEFT, idirl)*0.5;
	    } else {
            betal = _x.xsdf(ig, ll) / _g.hmesh(idirl, ll);
	    }
        if(lr < 0) {
            betar = _g.albedo(RIGHT, idirr)*0.5;
        } else {
            betar = _x.xsdf(ig,lr) / _g.hmesh(idirr, lr);
        }
        dtil(ig,ls)=2*betal*betar/(betal+betar);
	}
}

void CMFD::upddhat(const int &ls, SOL_VAR* flux, SOL_VAR* jnet) {
    int ll = _g.lklr(LEFT,ls);
    int lr = _g.lklr(RIGHT,ls);
    int idirl = _g.idirlr(LEFT,ls);
    int idirr = _g.idirlr(RIGHT,ls);

    for (int ig = 0; ig < _g.ng(); ig++)
    {
        if(ll < 0) {
            double jnet_fdm =-dtil(ig,ls)*(flux(ig,lr));
            dhat(ig,ls) = (jnet_fdm - jnet(ig,ls)) / (flux(ig,lr));
        } else if (lr < 0) {
            double jnet_fdm =-dtil(ig,ls)*(-flux(ig,ll));
            dhat(ig,ls) = (jnet_fdm - jnet(ig,ls)) / (flux(ig,ll));
        } else {
            double jnet_fdm =-dtil(ig,ls)*(flux(ig,lr)-flux(ig,ll));
            dhat(ig,ls) = (jnet_fdm - jnet(ig,ls)) / (flux(ig,lr)+flux(ig,ll));
        }
    }
}

void CMFD::setls(const int &l) {
    // determine the area of surfaces at coarse meshes that is normal to directions
    float area[NDIRMAX];

    area[XDIR] = _g.hmesh(YDIR, l) * _g.hmesh(ZDIR, l);
    area[YDIR] = _g.hmesh(XDIR, l) * _g.hmesh(ZDIR, l);
    area[ZDIR] = _g.hmesh(XDIR, l) * _g.hmesh(YDIR, l);

    for (int ige = 0; ige < _g.ng(); ++ige) {

        for (int igs = 0; igs < _g.ng(); ++igs) {
            diag(igs,ige,l) = -_x.xssf(igs, ige, l)*_g.vol(l);
        }
        diag(ige,ige,l) += _x.xstf(ige, l)*_g.vol(l);

        for (int idir = NDIRMAX - 1; idir >= 0; --idir)
        {
            int ln = _g.neib(LEFT, idir, l);

            if (ln >= 0) {
                int ls = _g.lktosfc(LEFT, idir, l);

                cc(LEFT,idir,ige,l) =(-dtil(ige, ls) + dhat(ige, ls))* area[idir];
                diag(ige,ige,l) += (dtil(ige, ls) + dhat(ige, ls)) * area[idir];
            }
        }

        for (int idir = 0; idir < NDIRMAX; idir++)
        {
            int ln = _g.neib(RIGHT, idir, l);

            if (ln >= 0) {
                int ls = _g.lktosfc(RIGHT, idir, l);
                cc(RIGHT,idir,ige,l) =(-dtil(ige, ls) - dhat(ige, ls))* area[idir];
                diag(ige,ige,l) += (dtil(ige, ls) - dhat(ige, ls)) * area[idir];
            }
        }
    }
}

void CMFD::setNcmfd(int ncmfd) {
    _ncmfd = ncmfd;
}

void CMFD::setEpsl2(float epsl2) {
    _epsl2 = epsl2;
}

void CMFD::updpsi(const int& l, const SOL_VAR* flux) {

    _psi[l] = 0.0;

    for (int ig = 0; ig < _g.ng(); ig++)
    {
        _psi[l] += flux(ig,l) * _x.xsnf(ig, l);
    }
    _psi[l] = _psi[l] * _g.vol(l);

    //printf("%d %e %e %e %e %e %e \n", l, _x.xsnf(0, l), _x.xsnf(1, l), flux(0, l), flux(1, l), _g.vol(l), _psi[l]);

}
