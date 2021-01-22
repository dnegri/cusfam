#include "CMFD.h"

#define flux(ig, l)   (flux[(l)*_g.ng()+ig])
#define jnet(ig, ls)      (jnet[(ls)*_g.ng() + ig])

CMFD::CMFD(Geometry &g, CrossSection& x) : _g(g), _x(x){
    _ls = new MKLSolver(g);
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

	for (size_t ig = 0; ig < _g.ng(); ig++)
	{
	    if(ll < 0) {
            betal = _g.albedo(LEFT, idirl);
	    } else {
            betal = _x.xsdf(ig, ll) / _g.hmesh(idirl, ll);
	    }
        if(lr < 0) {
            betar = _g.albedo(RIGHT, idirr);
        } else {
            betar = _x.xsdf(ig,lr) / _g.hmesh(idirr, lr);
        }
        dtil(ig,ls)=2*betal*betar/(betal+betar);
	}
}

void CMFD::upddhat(const int &ls, double* flux, float* jnet) {
    int ll = _g.lklr(LEFT,ls);
    int lr = _g.lklr(RIGHT,ls);
    int idirl = _g.idirlr(LEFT,ls);
    int idirr = _g.idirlr(RIGHT,ls);

    float jnet_fdm;

    for (size_t ig = 0; ig < _g.ng(); ig++)
    {
        jnet_fdm =-dtil(ig,ls)*(flux(ig,lr)-flux(ig,ll));
        dhat(ig,ls) = (jnet_fdm - jnet(ig,ls)) / (flux(ig,lr)+flux(ig,ll));
    }
}

void CMFD::setls(const int &l) {
    // determine the area of surfaces at coarse meshes that is normal to directions
    float area[NDIRMAX];

    area[XDIR] = _g.hmesh(YDIR, l) * _g.hmesh(ZDIR, l);
    area[YDIR] = _g.hmesh(XDIR, l) * _g.hmesh(ZDIR, l);
    area[ZDIR] = _g.hmesh(XDIR, l) * _g.hmesh(YDIR, l);

    float sgn[2]{1.0,-1.0};

    int idxrow = l * _g.ng();

    for (int ige = 0; ige < _g.ng(); ++ige) {
        int irow = _ls->indexRow(idxrow+ige);

        double diag = 0.0;
        for (size_t idir = NDIRMAX - 1; idir >= 0; --idir)
        {
            int ln = _g.neib(LEFT, idir, l);

            if (ln >= 0) {
                int ls = _g.lktosfc(LEFT, idir, l);
                _ls->a(irow++) = (-dtil(ige, ls) + dhat(ige, ls))* area[idir];
                diag += (dtil(ige, ls) + dhat(ige, ls)) * area[idir];
            }
        }

        //diagonal
        int idiag = irow + ige;
        _ls->a(idiag) = _x.xstf(ige, l);

        for (size_t igs = 0; igs < _g.ng(); igs++)
        {
            _ls->a(irow++) -= (_x.chif(ige, l) * _x.xsnf(igs, l) * _reigvs + _x.xssf(igs, ige, l)) * _g.vol(l);
        }

        for (size_t idir = 0; idir < NDIRMAX; idir++)
        {
            int ln = _g.neib(RIGHT, idir, l);

            if (ln >= 0) {
                int ls = _g.lktosfc(RIGHT, idir, l);
                _ls->a(irow++) = (-dtil(ige, ls) - dhat(ige, ls)) * area[idir];
                diag += (dtil(ige, ls) - dhat(ige, ls)) * area[idir];
            }
        }

        _ls->a(idiag) += diag;
    }
}


