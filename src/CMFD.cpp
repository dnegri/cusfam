#include "CMFD.h"


#define flux(ig, l)   (flux[(l)*_g.ng()+ig])
#define jnet(ig, ls)      (jnet[(ls)*_g.ng() + ig])

CMFD::CMFD(Geometry &g, CrossSection& x) : _g(g), _x(x){
    _dtil = new double[_g.nsurf() * _g.ng()]{};
    _dhat = new double[_g.nsurf() * _g.ng()]{};
    _diag = new double[_g.nxyz() * _g.ng2()]{};
    _cc = new double[_g.nxyz() * _g.ng() * NEWSBT]{};
    _src = new double[_g.nxyz() * _g.ng()]{};
    _eshift = 0.0;
    _epsl2 = 1.E-5;
}

CMFD::~CMFD() {

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

void CMFD::upddhat(const int &ls, double* flux, double* jnet) {
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

void CMFD::setEshift(float eshift) {
    _eshift = eshift;
}

void CMFD::setEpsl2(float epsl2) {
    _epsl2 = epsl2;
}


