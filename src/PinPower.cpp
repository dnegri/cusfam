#include "PinPower.h"

#define flux(ig, lk)   (flux[(lk)*_g.ng()+ig])
#define phis(ig, lks)  (phis[(lks)*_g.ng() + ig])
#define jnet(ig, lks)  (jnet[(lks)*_g.ng() + ig])

#define pinphili(ig, ipinxy,  jpinxy,  li) \
    pinphili[((li)*_npinxy*_npinxy+(jpinxy)*_npinxy+(ipinxy))*_g.ng()+ (ig)]

#define pinphisym(ig, ipinxy,  jpinxy) \
    pinphisym[((jpinxy)*_npinxy+(ipinxy))*_g.ng()+ (ig)]


PinPower::PinPower(Geometry & g, CrossSection& x) : _g(g), _x(x)
{
	int ng = g.ng();
	int ncorn = 0;
	int nz = g.nz();
	int nxy = g.nxy();

	_phicorn0 = new PPR_VAR[ng* ncorn* nz];
	_phicorn = new PPR_VAR[ng* ncorn* nz];
	_avgjnetz = new PPR_VAR[g.ngxyz()];
	_trlzcff = new PPR_VAR[9 * g.ngxyz()];
	_kappa = new PPR_VAR[g.ngxyz()];
	_qf2d = new PPR_VAR[nterm*g.ngxyz()];
	_qc2d = new PPR_VAR[nterm*g.ngxyz()];
	_pc2d = new PPR_VAR[nterm*g.ngxyz()];
	_hc2d = new PPR_VAR[8* g.ngxyz()];
	_jcornx = new PPR_VAR[4* g.ngxyz()];
	_jcorny = new PPR_VAR[4* g.ngxyz()];
	_clsqf01 = new PPR_VAR[g.ngxyz()];
	_clsqf02 = new PPR_VAR[g.ngxyz()];
	_clsqf11 = new PPR_VAR[g.ngxyz()];
	_clsqf12 = new PPR_VAR[g.ngxyz()];
	_clsqf21 = new PPR_VAR[g.ngxyz()];
	_clsqf22 = new PPR_VAR[g.ngxyz()];
	_clsqf31 = new PPR_VAR[g.ngxyz()];
	_clsqf32 = new PPR_VAR[g.ngxyz()];
	_clsqf41 = new PPR_VAR[g.ngxyz()];
	_clsqf42 = new PPR_VAR[g.ngxyz()];
	_clsqfx1y1 = new PPR_VAR[g.ngxyz()];
	_clsqf1221 = new PPR_VAR[g.ngxyz()];
	_clsqf1331 = new PPR_VAR[g.ngxyz()];
	_clsqfx2y2 = new PPR_VAR[g.ngxyz()];
	_cpc02 = new PPR_VAR[g.ngxyz()];
	_cpc04 = new PPR_VAR[g.ngxyz()];
	_cpc022 = new PPR_VAR[g.ngxyz()];
	_cpc11 = new PPR_VAR[g.ngxyz()];
	_cpc12 = new PPR_VAR[g.ngxyz()];
	_cpc21 = new PPR_VAR[g.ngxyz()];
	_cpc22 = new PPR_VAR[g.ngxyz()];
	_chc6 = new PPR_VAR[g.ngxyz()];
	_chc13j = new PPR_VAR[g.ngxyz()];
	_chc13p = new PPR_VAR[g.ngxyz()];
	_chc57j = new PPR_VAR[g.ngxyz()];
	_chc57p = new PPR_VAR[g.ngxyz()];
	_chc8j = new PPR_VAR[g.ngxyz()];
	_chc8p = new PPR_VAR[g.ngxyz()];
	_chc24j = new PPR_VAR[g.ngxyz()];
	_chc24a = new PPR_VAR[g.ngxyz()];
	_cpjxh1 = new PPR_VAR[4* g.ngxyz()];
	_cpjxh2 = new PPR_VAR[4* g.ngxyz()];
	_cpjxh5 = new PPR_VAR[4* g.ngxyz()];
	_cpjxh6 = new PPR_VAR[4* g.ngxyz()];
	_cpjxh7 = new PPR_VAR[4* g.ngxyz()];
	_cpjxh8 = new PPR_VAR[4* g.ngxyz()];
	_cpjxp6 = new PPR_VAR[4* g.ngxyz()];
	_cpjxp7 = new PPR_VAR[4* g.ngxyz()];
	_cpjxp8 = new PPR_VAR[4* g.ngxyz()];
	_cpjxp9 = new PPR_VAR[4* g.ngxyz()];
	_cpjxp11 = new PPR_VAR[4* g.ngxyz()];
	_cpjxp12 = new PPR_VAR[4* g.ngxyz()];
	_cpjxp13 = new PPR_VAR[4* g.ngxyz()];
	_cpjxp14 = new PPR_VAR[4* g.ngxyz()];
	_cpjyh3 = new PPR_VAR[4* g.ngxyz()];
	_cpjyh4 = new PPR_VAR[4* g.ngxyz()];
	_cpjyh5 = new PPR_VAR[4* g.ngxyz()];
	_cpjyh6 = new PPR_VAR[4* g.ngxyz()];
	_cpjyh7 = new PPR_VAR[4* g.ngxyz()];
	_cpjyh8 = new PPR_VAR[4* g.ngxyz()];
	_cpjyp2 = new PPR_VAR[4* g.ngxyz()];
	_cpjyp3 = new PPR_VAR[4* g.ngxyz()];
	_cpjyp4 = new PPR_VAR[4* g.ngxyz()];
	_cpjyp9 = new PPR_VAR[4* g.ngxyz()];
	_cpjyp10 = new PPR_VAR[4* g.ngxyz()];
	_cpjyp12 = new PPR_VAR[4* g.ngxyz()];
	_cpjyp13 = new PPR_VAR[4* g.ngxyz()];
	_cpjyp14 = new PPR_VAR[4* g.ngxyz()];

    _pinpowa = new PPR_VAR[g.ngxyz()*_g.npinxy()*_g.npinxy()]{};
    _pinphia = new PPR_VAR[g.ngxyz()*_g.npinxy()*_g.npinxy()]{};


    _nrest = _g.npinxy() % _g.ndivxy();
    _npex = _g.npinxy() / _g.ndivxy();
    _npinxy = _npex+_nrest;

    _hpini = new PPR_VAR[_npinxy*_g.ndivxy()*_g.ndivxy()];
    _hpinj = new PPR_VAR[_npinxy*_g.ndivxy()*_g.ndivxy()];
    _pcoeff = new PPR_VAR[nterm*_npinxy*_npinxy*_g.ndivxy()*_g.ndivxy()];

    double ml = 1/sqrt2;
    double vl[2], vl2[2], vr[2], vr2[2];
    double hpin0 = 2.*_g.ndivxy() / _g.npinxy();

    for (int j = 0; j < _g.ndivxy(); ++j) {
        for (int i = 0; i < _g.ndivxy(); ++i) {
            auto li = j*_g.ndivxy() + i;

            vl[YDIR]=-1.0;
            vl2[YDIR]=1.0;

            for (int jp = 0; jp < _npinxy; ++jp) {
                double hpin = hpin0;
                if(_nrest != 0 && ((jp == 0 && j==1) || (jp==_npinxy-1 && j == 0))) {
                    hpin = hpin * 0.5;
                }

                hpinj(jp,li) = hpin;
                vr[YDIR]=vl[YDIR]+hpin;
                vr2[YDIR]=vr[YDIR]*vr[YDIR];

                vl[XDIR]=-1;
                vl2[XDIR]=1;

                for (int ip = 0; ip < _npinxy; ++ip) {
                    hpin = hpin0;
                    if(_nrest != 0 && ((ip == 0 && i==1) || (ip==_npinxy-1 && i == 0))) {
                        hpin = hpin * 0.5;
                    }
                    hpini(ip,li)=hpin;
                    vr[XDIR]=vl[XDIR]+hpin;
                    vr2[XDIR]=vr[XDIR]*vr[XDIR];

                    int icf = -1;
                    for (int idir = YDIR; idir > -1; --idir) {
                        pcoeff(icf+1,ip,jp,li)=0.5*(vl[idir]+vr[idir]);
                        pcoeff(icf+2,ip,jp,li)=0.5*(-1.+vl2[idir]+vl[idir]*vr[idir]+vr2[idir]);
                        pcoeff(icf+3,ip,jp,li)=0.125*(vl[idir]+vr[idir])*(-6.+5.*(vl2[idir]+vr2[idir]));
                        pcoeff(icf+4,ip,jp,li)=0.125*(3.+7.*vl2[idir]*vr2[idir]-10.*vl[idir]*vr[idir]
                                                      -10.*vl2[idir]+7.*vl2[idir]*vl2[idir]+7.*vl2[idir]*vl[idir]*vr[idir]
                                                      -10.*vr2[idir]+7.*vr2[idir]*vr2[idir]+7.*vr2[idir]*vr[idir]*vl[idir]);
                        icf = icf + 4;

                    }
                    pcoeff( 9,ip,jp,li) = pcoeff(1,ip,jp,li) * pcoeff(5,ip,jp,li);
                    pcoeff(10,ip,jp,li) = pcoeff(2,ip,jp,li) * pcoeff(5,ip,jp,li);
                    pcoeff(11,ip,jp,li) = pcoeff(1,ip,jp,li) * pcoeff(6,ip,jp,li);
                    pcoeff(12,ip,jp,li) = pcoeff(2,ip,jp,li) * pcoeff(6,ip,jp,li);
                    pcoeff(13,ip,jp,li) = 0.25*pcoeff( 9,ip,jp,li)*(-6+5*(vl2[YDIR]+vr2[YDIR]));
                    pcoeff(14,ip,jp,li) = 0.25*pcoeff( 9,ip,jp,li)*(-6+5*(vl2[XDIR]+vr2[XDIR]));

                    vl[XDIR] =vr[XDIR];
                    vl2[XDIR]=vr2[XDIR];
                }
                vl[YDIR] =vr[YDIR];
                vl2[YDIR]=vr2[YDIR];

            }
        }
    }
}

PinPower::~PinPower()
{
}

void PinPower::calphicorn(SOL_VAR* flux, SOL_VAR* phis)
{
	for (int k = 0; k < _g.nz(); k++)
	{
		for (int lc = 0; lc < _g.ncorn(); lc++)
		{
			for (int ig = 0; ig < _g.ng(); ig++)
			{
				phicorn(ig, lc, k) = 0.0;
				int nodecnt = 0;

				int l = _g.lctol(NW, lc);
				int lk = k * _g.nxy() + l;

				if (l != -1) {
					int lk_x = _g.lktosfc(RIGHT, XDIR, lk);
					int lk_y = _g.lktosfc(RIGHT, YDIR, lk);

					nodecnt = nodecnt + 1;
					phicorn(ig, lc, k) +=
						phis(ig,lk_x) + phis(ig,lk_y) - flux(ig, lk);

				}

				l = _g.lctol(NE, lc);
				lk = k * _g.nxy() + l;
				if (l != -1) {
					int lk_x = _g.lktosfc(LEFT, XDIR, lk);
					int lk_y = _g.lktosfc(RIGHT, YDIR, lk);

					nodecnt = nodecnt + 1;
					phicorn(ig, lc, k) +=
						phis(ig, lk_x) + phis(ig, lk_y) - flux(ig, lk);

				}

				l = _g.lctol(SE, lc);
				lk = k * _g.nxy() + l;
				if (l != -1) {
					int lk_x = _g.lktosfc(LEFT, XDIR, lk);
					int lk_y = _g.lktosfc(LEFT, YDIR, lk);

					nodecnt = nodecnt + 1;
					phicorn(ig, lc, k) +=
						phis(ig, lk_x) + phis(ig, lk_y) - flux(ig, lk);

				}

				l = _g.lctol(SW, lc);
				lk = k * _g.nxy() + l;
				if (l != -1) {
					int lk_x = _g.lktosfc(RIGHT, XDIR, lk);
					int lk_y = _g.lktosfc(LEFT, YDIR, lk);

					nodecnt = nodecnt + 1;
					phicorn(ig, lc, k) +=
						phis(ig, lk_x) + phis(ig, lk_y) - flux(ig, lk);
				}

				phicorn(ig, lc, k) /= nodecnt;

				if (phicorn(ig, lc, k)<0.0) {
					PLOG(plog::error) << "There is negative corner flux.";
					exit(-1);
				}

			}
		}
	}
}

void PinPower::calhomo(const double& eigv, SOL_VAR* flux, SOL_VAR* phis, SOL_VAR* jnet) {

    for (int k = 0; k < _g.nz(); ++k) {
        for (int l = 0; l < _g.nxy(); ++l) {
            calcff(l,k);
            expflux13(l,k,flux,phis,jnet);
            calsol2drhs(l, k, eigv);
            calsol(l,k,jnet);
        }
    }
}

void PinPower::caltrlz(int l, int k, SOL_VAR* jnet) {
    const static auto r4=0.25;
    const static auto r12=1./12.0;
    const static auto r16=0.25*0.25;
    const static auto r48=1./48.0;
    const static auto r144=1./144.0;

    auto lk = k*_g.nxy()+l;
    auto rhz = 1/_g.hmesh(ZDIR,lk);

    for (int ig = 0; ig < _g.ng(); ++ig) {
        int lksl = _g.lktosfc(LEFT, ZDIR, lk);
        int lksr = _g.lktosfc(RIGHT, ZDIR, lk);

        auto ajnetz =(jnet(ig,lksr)-jnet(ig,lksl))*rhz;
        trlzcff(0,0,ig,lk)  =ajnetz;
    }

}

void PinPower::expflux13(int l, int k, SOL_VAR* flux, SOL_VAR* phis, SOL_VAR* jnet) {
    const static double r14 = 1./14.0;

    int lk = _g.nxy()*k + l;
    for (int ig = 0; ig < _g.ng(); ++ig) {
        auto rxsdf=1/_x.xsdf(ig,lk);
        auto alpha=-0.5*_g.hmesh(0,lk)*rxsdf;

        PPR_VAR ps[2], pd[2], js[2], jd[2];
        for (int idir = 0; idir < 2; ++idir) {
            int lksl = _g.lktosfc(LEFT, idir, lk);
            int lksr = _g.lktosfc(RIGHT, idir, lk);

            ps[idir] = phis(ig, lksr) + phis(ig, lksl);
            pd[idir] = phis(ig, lksr) - phis(ig, lksl);
            js[idir] = alpha * (jnet(ig, lksr) + jnet(ig, lksl));
            jd[idir] = alpha * (jnet(ig, lksr) - jnet(ig, lksl));
        }

        auto val1=phicorn(ig,_g.ltolc(NW,lk),k);          //  1(nw)-----2(ne)
        auto val2=phicorn(ig,_g.ltolc(NE,lk),k);          //    |         |
        auto val3=phicorn(ig,_g.ltolc(SW,lk),k);          //    |   node  |
        auto val4=phicorn(ig,_g.ltolc(SE,lk),k);          //    |         |
        auto p1=0.25*(val4-val3+val2-val1);                           //  3(sw)-----4(se)
        auto p2=0.25*(val4-val3-val2+val1);
        auto p3=0.25*(val4+val3-val2-val1);
        auto p4=0.25*(val4+val3+val2+val1);

        qf2d(0,ig,lk)=flux(ig,lk);

        for (int idir = 0; idir < 2; ++idir) {
            auto io = 1;
            if (idir == XDIR) io = 5;
            auto temp = ps[idir] - 2 * flux(ig, lk);
            qf2d(io + 0, ig, lk) = 0.1 * (-js[idir] + 6 * pd[idir]);        //5- y^1, 1- x^1
            qf2d(io + 1, ig, lk) = r14 * (-jd[idir] + 10 * temp);           //6- y^2, 2- x^2
            qf2d(io + 2, ig, lk) = 0.1 * (+js[idir] - pd[idir]);          //7- y^3, 3- x^3
            qf2d(io + 3, ig, lk) = r14 * (+jd[idir] - 3 * temp);            //8- y^4, 4- x^4
        }
        qf2d(9,ig,lk)=p2;                                   // 9- x^1 * y^1
        qf2d(10,ig,lk)=p1-0.5*pd[0];                       //10- x^1 * y^2
        qf2d(11,ig,lk)=p3-0.5*pd[1];                       //11- x^2 * y^1
        qf2d(12,ig,lk)=flux(ig,lk)+p4-0.5*(ps[0]+ps[1]);  //2- x^2 * y^2
    }
}

void PinPower::calsol2drhs(int l, int k, const double & reigv ) {

    int lk = k*_g.nxy()+l;
    for (int i = 0; i < nterm; ++i) {
        double psi2d=0.0;
        double sc2d[2]{};

        for (int ig = 0; ig < _g.ng(); ++ig) {
            psi2d=psi2d+reigv*_x.xsnf(ig,lk)*qf2d(i,ig,lk);
            for (int igs = 0; igs < _g.ng(); ++igs) {
                sc2d[ig]=sc2d[ig]+_x.xssf(igs,ig,lk)*qf2d(i,igs,lk);
            }
        }
        for (int ig = 0; ig < _g.ng(); ++ig) {
            qc2d(i,ig,lk)=_x.chif(ig,lk)*psi2d+sc2d[ig];
        }
    }

    // 1-(0,1), 2-(0,2), 3-(0,3), 4-(0,4)
    // 5-(1,0), 6-(2,0), 7-(3,0), 8-(4,0)
    // 9-(1,1), 10-(1,2),11-(2,1),12-(2,2)
    for (int ig = 0; ig < _g.ng(); ++ig) {
        qc2d(0,ig,lk)=qc2d(0,ig,lk)-trlzcff(0,0,ig,lk);
        qc2d(1,ig,lk)=qc2d(1,ig,lk)-trlzcff(0,1,ig,lk);
        qc2d(2,ig,lk)=qc2d(2,ig,lk)-trlzcff(0,2,ig,lk);
        qc2d(5,ig,lk)=qc2d(5,ig,lk)-trlzcff(1,0,ig,lk);
        qc2d(6,ig,lk)=qc2d(6,ig,lk)-trlzcff(2,0,ig,lk);
        qc2d(9,ig,lk)=qc2d(9,ig,lk)-trlzcff(1,1,ig,lk);
        qc2d(10,ig,lk)=qc2d(10,ig,lk)-trlzcff(1,2,ig,lk);
        qc2d(11,ig,lk)=qc2d(11,ig,lk)-trlzcff(2,1,ig,lk);
        qc2d(12,ig,lk)=qc2d(12,ig,lk)-trlzcff(2,2,ig,lk);
    }
}

void PinPower::calsol(int l, int k, SOL_VAR * jnet) {

    auto lk = k*_g.nxy()+l;
    auto rh = 1./_g.hmesh(XDIR,lk);
    for (int ig = 0; ig < _g.ng(); ++ig) {

        auto rxstf = 1. / _x.xstf(ig, lk);

        //  for pc2d(2, ig, lk) and pc2d(6, ig, lk);
        auto cpc22qc2d12 = cpc22(ig, lk) * qc2d(12, ig, lk);

        // coefficients of particular solutions
        //1 - (0, 1), 2 - (0, 2), 3 - (0, 3), 4 - (0, 4);
        //5 - (1, 0), 6 - (2, 0), 7 - (3, 0), 8 - (4, 0);
        //9 - (1, 1), 10 - (1, 2), 11 - (2, 1), 12 - (2, 2);
        pc2d(0, ig, lk) = cpc02(ig, lk) * (qc2d(2, ig, lk) + qc2d(6, ig, lk))
                          + cpc04(ig, lk) * (qc2d(4, ig, lk) + qc2d(8, ig, lk))
                          + cpc022(ig, lk) * qc2d(12, ig, lk)
                          + rxstf * qc2d(0, ig, lk);

        pc2d(1, ig, lk) = cpc11(ig, lk) * qc2d(3, ig, lk)
                          + cpc12(ig, lk) * qc2d(11, ig, lk)
                          + rxstf * qc2d(1, ig, lk);

        pc2d(2, ig, lk) = cpc21(ig, lk) * qc2d(4, ig, lk)
                          + cpc22qc2d12
                          + rxstf * qc2d(2, ig, lk);

        pc2d(3, ig, lk) = rxstf * qc2d(3, ig, lk);

        pc2d(4, ig, lk) = rxstf * qc2d(4, ig, lk);

        pc2d(5, ig, lk) = cpc11(ig, lk) * qc2d(7, ig, lk)
                          + cpc12(ig, lk) * qc2d(10, ig, lk)
                          + rxstf * qc2d(5, ig, lk);

        pc2d(6, ig, lk) = cpc21(ig, lk) * qc2d(8, ig, lk)
                          + cpc22qc2d12
                          + rxstf * qc2d(6, ig, lk);

        pc2d(7, ig, lk) = rxstf * qc2d(7, ig, lk);

        pc2d(8, ig, lk) = rxstf * qc2d(8, ig, lk);

        pc2d(9, ig, lk) = rxstf * qc2d(9, ig, lk)
                          + 60 * _x.xsdf(ig, lk) * rxstf * rxstf * rh * rh
                            * (qc2d(13, ig, lk) + qc2d(14, ig, lk));

        pc2d(10, ig, lk) = rxstf * qc2d(10, ig, lk);

        pc2d(11, ig, lk) = rxstf * qc2d(11, ig, lk);

        pc2d(12, ig, lk) = rxstf * qc2d(12, ig, lk);

        if (term15) {
            pc2d(13, ig, lk) = rxstf * qc2d(13, ig, lk);
            pc2d(14, ig, lk) = rxstf * qc2d(14, ig, lk);
        }

        // obtain corner points for homogenous boundary conditions
        auto p0yp1p1 = +pc2d(1, ig, lk) + pc2d(2, ig, lk) + pc2d(3, ig, lk) + pc2d(4, ig, lk);
        auto p0ym1p1 = -pc2d(1, ig, lk) + pc2d(2, ig, lk) - pc2d(3, ig, lk) + pc2d(4, ig, lk);
        auto px0p1p1 = +pc2d(5, ig, lk) + pc2d(6, ig, lk) + pc2d(7, ig, lk) + pc2d(8, ig, lk);
        auto px0m1p1 = -pc2d(5, ig, lk) + pc2d(6, ig, lk) - pc2d(7, ig, lk) + pc2d(8, ig, lk);

        auto pxyp1p1 =
                +pc2d(9, ig, lk) + pc2d(10, ig, lk) + pc2d(11, ig, lk) + pc2d(12, ig, lk) + pc2d(13, ig, lk) +
                pc2d(14, ig, lk);
        auto pxym1m1 =
                -pc2d(9, ig, lk) - pc2d(10, ig, lk) + pc2d(11, ig, lk) + pc2d(12, ig, lk) - pc2d(13, ig, lk) -
                pc2d(14, ig, lk);
        auto pxym1p1 =
                -pc2d(9, ig, lk) + pc2d(10, ig, lk) - pc2d(11, ig, lk) + pc2d(12, ig, lk) - pc2d(13, ig, lk) -
                pc2d(14, ig, lk);
        auto pxyp1m1 =
                +pc2d(9, ig, lk) - pc2d(10, ig, lk) - pc2d(11, ig, lk) + pc2d(12, ig, lk) + pc2d(13, ig, lk) +
                pc2d(14, ig, lk);

        //x = -1, y = -1
        auto fm1m1 = phicorn(ig, _g.ltolc(NW, l), k) - (pc2d(0, ig, lk) + p0ym1p1 + px0m1p1 + pxyp1m1);
        //x = +1, y = -1
        auto fp1m1 = phicorn(ig, _g.ltolc(NE, l), k) - (pc2d(0, ig, lk) + p0ym1p1 + px0p1p1 + pxym1p1);
        //x = -1, y = +1
        auto fm1p1 = phicorn(ig, _g.ltolc(SW, l), k) - (pc2d(0, ig, lk) + p0yp1p1 + px0m1p1 + pxym1m1);
        //x = +1, y = +1
        auto fp1p1 = phicorn(ig, _g.ltolc(SE, l), k) - (pc2d(0, ig, lk) + p0yp1p1 + px0p1p1 + pxyp1p1);

        auto p1 = 0.25 * (fp1p1 - fm1p1 + fp1m1 - fm1m1);
        auto p2 = 0.25 * (fp1p1 - fm1p1 - fp1m1 + fm1m1);
        auto p3 = 0.25 * (fp1p1 + fm1p1 - fp1m1 - fm1m1);
        auto p4 = 0.25 * (fp1p1 + fm1p1 + fp1m1 + fm1m1);

        //obtain jnet for homegeneous boundary condtions
        auto alpha = -2 * _x.xsdf(ig, lk) * rh;

        auto lkxl = _g.lktosfc(LEFT, XDIR, lk);
        auto lkxr = _g.lktosfc(RIGHT, XDIR, lk);
        auto lkyl = _g.lktosfc(LEFT, YDIR, lk);
        auto lkyr = _g.lktosfc(RIGHT, YDIR, lk);

        double jhr[2], jhl[2], jhs[2], jhd[2];

        jhr[XDIR] = jnet(ig, lkxr) -
                 alpha * (pc2d(5, ig, lk) + 3 * pc2d(6, ig, lk) + 6 * pc2d(7, ig, lk) + 10 * pc2d(8, ig, lk));
        jhl[XDIR] = jnet(ig, lkxl) -
                 alpha * (pc2d(5, ig, lk) - 3 * pc2d(6, ig, lk) + 6 * pc2d(7, ig, lk) - 10 * pc2d(8, ig, lk));
        jhr[YDIR] = jnet(ig, lkyr) -
                 alpha * (pc2d(1, ig, lk) + 3 * pc2d(2, ig, lk) + 6 * pc2d(3, ig, lk) + 10 * pc2d(4, ig, lk));
        jhl[YDIR] = jnet(ig, lkyl) -
                 alpha * (pc2d(1, ig, lk) - 3 * pc2d(2, ig, lk) + 6 * pc2d(3, ig, lk) - 10 * pc2d(4, ig, lk));

        auto temp = 0.5 / kappa(ig, lk);
        for (int idir = 0; idir < 2; ++idir) {
            jhs[idir] = temp * (jhr[idir] + jhl[idir]);
            jhd[idir] = temp * (jhr[idir] - jhl[idir]);
        }

        // coefficients of homogeneous solutions
        hc2d(6, ig, lk) = chc6(ig, lk) * p2;
        hc2d(1, ig, lk) = chc13j(ig, lk) * jhs[XDIR] + chc13p(ig, lk) * p1;
        hc2d(3, ig, lk) = chc13j(ig, lk) * jhs[YDIR] + chc13p(ig, lk) * p3;
        hc2d(5, ig, lk) = chc57j(ig, lk) * jhs[XDIR] + chc57p(ig, lk) * p1;
        hc2d(7, ig, lk) = chc57j(ig, lk) * jhs[YDIR] + chc57p(ig, lk) * p3;
        hc2d(8, ig, lk) = chc8j(ig, lk) * (jhd[XDIR] + jhd[YDIR]) + chc8p(ig, lk) * p4;
        auto chc24hc2d8 = chc24a(ig, lk) * hc2d(8, ig, lk);
        hc2d(2, ig, lk) = chc24j(ig, lk) * jhd[XDIR] + chc24hc2d8;
        hc2d(4, ig, lk) = chc24j(ig, lk) * jhd[YDIR] + chc24hc2d8;
    }
}

void PinPower::calcff(int l, int k) {
    // copied from callcffsol2d

    // for coefficients of particular solutions
    // square cell only

    auto lk = k*_g.nxy()+l;

    // rh
    auto rh2=1/(_g.hmesh(XDIR,lk)*_g.hmesh(XDIR,lk));

    for (int ig = 0; ig < _g.ng(); ++ig) {
        kappa(ig,lk)=0.5*_g.hmesh(XDIR,lk)*sqrt(_x.xstf(ig,lk)/_x.xsdf(ig,lk));

        // rxstf
        auto sigr = _x.xstf(ig, lk);
        auto rsigr = 1. / sigr;
        auto rsigr2 = rsigr * rsigr;

        // rh*rh*rxstf*rxstf
        auto rh2rsigr2 = rh2 * rsigr2;

        // xsdf*xsdf
        auto sigdc = _x.xsdf(ig, lk);
        auto rsigdc = 1 / sigdc;
        auto rsigdc2 = rsigdc * rsigdc;

        // constant variable for coefficients of particular solutions
        cpc02(ig, lk) = 12 * sigdc * rh2rsigr2;
        cpc04(ig, lk) = 4 * sigdc * rh2rsigr2 * (420 * sigdc * rh2 * rsigr + 10);
        cpc022(ig, lk) = 288 * sigdc * sigdc * rh2rsigr2 * rh2 * rsigr;
        auto temp = 12 * sigdc * rh2rsigr2;
        cpc11(ig, lk) = 5 * temp;
        cpc12(ig, lk) = temp;

        temp = 4 * sigdc * rh2rsigr2;
        cpc21(ig, lk) = 35 * temp;
        cpc22(ig, lk) = 3 * temp;

        // for coefficients of homogeneous solutions
        auto rbk = 1 / kappa(ig,lk);
        auto cko = cosh(kappa(ig,lk));          // kappa-Original
        auto sko = sqrt(cko * cko - 1);
        auto rsko = 1 / sko;
        auto ckt = cosh(kappa(ig,lk) * rsqrt2);   // kappa-Tilda
        auto skt = sqrt(ckt * ckt - 1);

        auto ckt2 = ckt * ckt;
        auto skt2 = skt * skt;
        auto rxt = 1 / (skt * ckt);
        auto rbkckosko = 1 / (kappa(ig,lk) * cko - sko);
        auto bkh = kappa(ig,lk) * _g.hmesh(XDIR, lk);
        auto r2rsigdc = 0.5 * rsigdc;

        chc6(ig, lk) = 1 / skt2;
        chc13j(ig, lk) = bkh * r2rsigdc * (-rbkckosko);
        chc13p(ig, lk) = -rbkckosko;
        chc57j(ig, lk) = bkh * sko * r2rsigdc * rbkckosko * rxt;
        chc57p(ig, lk) = kappa(ig,lk) * cko * rbkckosko * rxt;
        temp = kappa(ig,lk) * 1 / (kappa(ig,lk) * sko * ckt2 - 2 * cko * skt2);
        chc8j(ig, lk) = temp * cko * _g.hmesh(XDIR, lk) * 0.5 * rsigdc;
        chc8p(ig, lk) = temp * sko;
        chc24j(ig, lk) = -_g.hmesh(XDIR, lk) * r2rsigdc * rsko;
        chc24a(ig, lk) = -skt2 * rsko * rbk;
    }
}

void PinPower::calpinpower(const int& la, const int& k, const int* larot1a) {

    const static int lindx[4]{0,1,3,2};
    const static int lirot[6]{2,4,3,1,2,4};

    double ml = 1/sqrt2;
    double vl[2], vl2[2], vr[2], vr2[2];
    double hpin0 = 2.*_g.ndivxy() / _g.npinxy();


    double * pinphili = new PPR_VAR[_g.ng()*_npinxy*_npinxy*_g.ndivxy()*_g.ndivxy()];
    double * pinphisym = new PPR_VAR[_g.ng()*_npinxy*_npinxy];

    for (int li = 0; li < _g.ndivxy() * _g.ndivxy(); ++li) {
        int liorg = -1;
        switch (larot1a[li]) {
            case 1:
            case 2:
            case 3:
                liorg = lirot[lindx[li]+larot1a[li]];
                break;
            case 11:
                liorg = li -1;
                break;
            case 12:
                liorg = li +1;
                break;
            case 13:
                liorg = li -2;
                break;
            case 14:
                liorg = li +2;
                break;
            case 0:
                liorg = li;
                break;
        }
        int l  = _g.latol(li,la);
        int lk = k*_g.nxy()+l;

        vl[YDIR]=-1;
        vl2[YDIR]=1;

        for (int jp = 0; jp < _npinxy; ++jp) {
            auto hpin = hpinj(jp, liorg);
            vr[YDIR] = vl[YDIR]+hpin;
            vr2[YDIR] = vr[YDIR]*vr[YDIR];

            vl[XDIR]=-1;
            vl2[XDIR]=1;
            for (int ip = 0; ip < _npinxy; ++ip) {
                hpin = hpini(ip,liorg);
                vr[XDIR]=vl[XDIR]+hpin;
                vr2[XDIR]=vr[XDIR]*vr[XDIR];
                for (int ig = 0; ig < _g.ng(); ++ig) {
                    PPR_VAR rdelt[2];
                    rdelt[XDIR]=1/((vl[XDIR]-vr[XDIR])*kappa(ig,lk));
                    rdelt[YDIR]=1/((vl[YDIR]-vr[YDIR])*kappa(ig,lk));

                    auto sinhxl=sinh(kappa(ig,lk)*vl[XDIR]);
                    auto sinhxr=sinh(kappa(ig,lk)*vr[XDIR]);
                    auto coshxl=sqrt(sinhxl*sinhxl+1);
                    auto coshxr=sqrt(sinhxr*sinhxr+1);

                    auto sinhyl=sinh(kappa(ig,lk)*vl[YDIR]);
                    auto sinhyr=sinh(kappa(ig,lk)*vr[YDIR]);
                    auto coshyl=sqrt(sinhyl*sinhyl+1);
                    auto coshyr=sqrt(sinhyr*sinhyr+1);

                    PPR_VAR hcoeff[8], coshm[2], sinhm[2];

                    hcoeff[0]=(coshxl-coshxr)*rdelt[XDIR];
                    hcoeff[1]=(sinhxl-sinhxr)*rdelt[XDIR];
                    hcoeff[2]=(coshyl-coshyr)*rdelt[YDIR];
                    hcoeff[3]=(sinhyl-sinhyr)*rdelt[YDIR];
                    auto denom = rdelt[XDIR]*rdelt[YDIR]*2;
                    auto kl=kappa(ig,lk)*ml;

                    for (int idir = XDIR; idir <= YDIR; ++idir) {
                        coshm[idir]=cosh(kl*vl[idir])-cosh(kl*vr[idir]);
                        sinhm[idir]=sinh(kl*vl[idir])-sinh(kl*vr[idir]);
                    }

                    hcoeff[4]=denom*coshm[XDIR]*sinhm[YDIR];
                    hcoeff[5]=denom*coshm[XDIR]*coshm[YDIR];
                    hcoeff[6]=denom*sinhm[XDIR]*coshm[YDIR];
                    hcoeff[7]=denom*sinhm[XDIR]*sinhm[YDIR];

                    auto phi1=pc2d(0,ig,lk);

                    for (int icf = 0; icf < nterm-1; ++icf) {
                        phi1=phi1+pcoeff(icf,ip,jp,liorg) *pc2d(icf,ig,lk);
                    }

                    for (int icf = 0; icf < 8; ++icf) {
                        phi1=phi1+hcoeff[icf]*hc2d(icf,ig,lk);
                    }

                    pinphili(ig,ip,jp,li) = phi1;
                }
                vl[XDIR] =vr[XDIR];
                vl2[XDIR]=vr2[XDIR];
            }
            vl[YDIR] =vr[YDIR];
            vl2[YDIR]=vr2[YDIR];
        }
        // handling symmetry
        switch(larot1a[li]) {
            case 0:
//                for (int jp = 0; jp < _npinxy; ++jp) {
//                    for (int ip = 0; ip < _npinxy; ++ip) {
//                        for (int ig = 0; ig < _g.ng(); ++ig) {
//                            pinphisym(ig, ip, jp) = pinphili(ig, ip, jp, li);
//                        }
//                    }
//                }
                break;
            case 1:
                for (int jp = 0; jp < _npinxy; ++jp) {
                    for (int ip = 0; ip < _npinxy; ++ip) {
                        for (int ig = 0; ig < _g.ng(); ++ig) {
                            pinphisym(ig, jp, _npinxy - ip + 1) = pinphili(ig, ip, jp, li);
                        }
                    }
                }
                break;
            case 2:
                for (int jp = 0; jp < _npinxy; ++jp) {
                    for (int ip = 0; ip < _npinxy; ++ip) {
                        for (int ig = 0; ig < _g.ng(); ++ig) {
                            pinphisym(ig, _npinxy - ip + 1, _npinxy - jp + 1) = pinphili(ig, ip, jp, li);
                        }
                    }
                }
                break;
            case 3:
                for (int jp = 0; jp < _npinxy; ++jp) {
                    for (int ip = 0; ip < _npinxy; ++ip) {
                        for (int ig = 0; ig < _g.ng(); ++ig) {
                            pinphisym(ig, _npinxy - jp + 1, ip) = pinphili(ig, ip, jp, li);
                        }
                    }
                }
                break;
            case 11:
            case 12:
                for (int jp = 0; jp < _npinxy; ++jp) {
                    for (int ip = 0; ip < _npinxy; ++ip) {
                        for (int ig = 0; ig < _g.ng(); ++ig) {
                            pinphisym(ig, _npinxy - ip + 1, jp) = pinphili(ig, ip, jp, li);
                        }
                    }
                }
                break;
            case 13:
            case 14:
                for (int jp = 0; jp < _npinxy; ++jp) {
                    for (int ip = 0; ip < _npinxy; ++ip) {
                        for (int ig = 0; ig < _g.ng(); ++ig) {
                            pinphisym(ig, ip, _npinxy - jp + 1) = pinphili(ig, ip, jp, li);
                        }
                    }
                }
                break;
        }
        if(larot1a[li] != 0) {
            for (int jp = 0; jp < _npinxy; ++jp) {
                for (int ip = 0; ip < _npinxy; ++ip) {
                    for (int ig = 0; ig < _g.ng(); ++ig) {
                        pinphili(ig, ip, jp, li) = pinphisym(ig, ip, jp);
                    }
                }
            }
        }
    }

    // calculate assemblywise pin flux
//    pinpowa(:,:,:) = 0
//    pinphia(:,:,:) = 0
    for (int j = 0; j < _g.ndivxy(); ++j) {
        auto jps = j*_npinxy-j*_nrest;
        for (int i = 0; i < _g.ndivxy(); ++i) {
            auto li = j*_g.ndivxy()+i;
            auto l = _g.latol(li, la);
            auto lk = k*_g.nxy()+l;
            auto ips = i*_npinxy-i*_nrest;
            for (int jp = 0; jp < _npinxy; ++jp) {
                auto jpa=jps+jp;
                for (int ip = 0; ip < _npinxy; ++ip) {
                    auto ipa=ips+ip;

                    for (int ig = 0; ig < _g.ng(); ++ig) {
                        pinpowa(ig,ipa,jpa,la,k)=pinpowa(ig,ipa,jpa,la,k)+pinphili(ig,ip,jp,li)*_x.xskf(ig,lk);
                        pinphia(ig,ipa,jpa,la,k)=pinphia(ig,ipa,jpa,la,k)+pinphili(ig,ip,jp,li);
                    }
                }
            }
        }
    }

    //if pin flux and power on the center line have to be averaged.
    if(_nrest != 0) {
        for (int ip = 0; ip < _g.npinxy(); ++ip) {
            for (int ig = 0; ig < _g.ng(); ++ig) {
                pinpowa(ig,ip,_npinxy,la,k)=pinpowa(ig,ip,_npinxy,la,k)*0.5;
                pinpowa(ig,_npinxy,ip,la,k)=pinpowa(ig,_npinxy,ip,la,k)*0.5;

                pinphia(ig,ip,_npinxy,la,k)=pinphia(ig,ip,_npinxy,la,k)*0.5;
                pinphia(ig,_npinxy,ip,la,k)=pinphia(ig,_npinxy,ip,la,k)*0.5;
            }    
        }
    }

    delete[] pinphili;
    delete[] pinphisym;


}

void PinPower::calpinpower() {
    for (int k = 0; k < _g.nz(); ++k) {
        for (int la = 0; la < _g.nxya(); ++la) {
            calpinpower(la,k,_g.larot(la));
        }
    }
}
