#include "CMFD.h"

#define flux(ig, l)   (flux[(l)*_g.ng()+ig])
#define jnet(ig, ls)      (jnet[(ls)*_g.ng() + ig])

CMFD::CMFD(Geometry &g, CrossSection& x) : _g(g), _x(x){

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

void CMFD::upddhat(const int &ls, float* flux, float* jnet) {
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

    for (int ig = 0; ig < _g.ng(); ++ig) {
        am(ig,ig,l) = _x.xstf(ig,l)  - _x.xsnf(ig,l) * _g.vol(l)*_x.chif(ig,l)*_reigvs;
        for (int idir = 0; idir < NDIRMAX; ++idir) {
            for (int lr = 0; lr < LR; ++lr) {
                int ls = _g.lktosfc(lr, idir, l);
                float offdiag=dtil(ig,ls)*area[idir];
                cc(lr,idir,ig,l) = -offdiag;
                am(ig,ig,l) = offdiag;
                offdiag=sgn[lr]*dhat(ig,ls)*area[idir];
                cc(lr,idir,ig,l) += offdiag;
                am(ig,ig,l) += offdiag;
            }
        }
    }
    af(0,l) = _x.xsnf(0,l) * _g.vol(l) * _x.chif(1,l);
    af(1,l) = _x.xsnf(1,l) * _g.vol(l) * _x.chif(0,l);

    am(1,0,l) = -_x.xssf(1,0,l)*_g.vol(l) - af(1,l)*_reigvs;
    am(0,1,l) = -_x.xssf(0,1,l)*_g.vol(l) - af(0,l)*_reigvs;
}


