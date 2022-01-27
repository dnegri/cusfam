#include "PinPower.h"

#define flux(ig, lk)   (flux[(lk)*_g.ng()+ig])
#define phis(ig, lks)  (phis[(lks)*_g.ng() + ig])

PinPower::PinPower(Geometry & g) : _g(g)
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
	_qf2d = new PPR_VAR[15*g.ngxyz()];
	_qc2d = new PPR_VAR[15*g.ngxyz()];
	_pc2d = new PPR_VAR[15*g.ngxyz()];
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
