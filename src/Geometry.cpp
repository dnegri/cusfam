#include "Geometry.h"

Geometry::Geometry()
{
}

Geometry::~Geometry()
{
}

void Geometry::setBoundaryCondition(int* symopt, int* symang, float* albedo)
{
	_symopt = *symopt;
	_symang = *symang;
	_albedo = new GEOM_VAR[LR*NDIRMAX];
	for (int idir = 0; idir < NDIRMAX; idir++)
	{
		for (int l = 0; l < LR; l++)
		{
			_albedo[idir * LR + l] = albedo[idir * LR + l];
		}
	}

	_part = _symang / 360.0;
}

void Geometry::initDimension(int* ng_, int* nxy_, int* nz_, int* nx_, int* ny_, int* nsurf_) {
	_ng = *ng_;
	_nx = *nx_;
	_ny = *ny_;
	_nz = *nz_;
	_nxy = *nxy_;
	_nxyz = _nxy * _nz;
	_nsurf = *nsurf_;
	_ng2 = _ng * _ng;
	_ngxyz = _nxyz * _ng;
	_ngxy = _nxy * _ng;
	_kbc = 1;
	_kec = _nz - 1;
}

void Geometry::initIndex(int* nxs_, int* nxe_, int* nys_, int* nye_, int * ijtol_, int * rotflg_,int* neibr_, float* hmesh_)
{
	_neibr = new int[_nxy*NEWS];
	_ijtol = new int[_nx*_ny];
    _rotflg = new int[_nx*_ny];
	_nxs = new int[_ny];
	_nxe = new int[_ny];
	_nys = new int[_nx];
	_nye = new int[_nx];

	_comps = new int[_nxyz];

	for (int j = 0; j < _ny; j++)
	{
		nxs(j) = nxs_[j]-1;
		nxe(j) = nxe_[j];
	}
	for (int i = 0; i < _nx; i++)
	{
		nys(i) = nys_[i] - 1;
		nye(i) = nye_[i];
	}

	for (int l = 0; l < _nxy; l++)
	{
		auto lnews0 = l * NEWS;
		for (int i4 = 0; i4 < NEWS; i4++)
		{
			neibr(i4,l) = neibr_[lnews0 + i4] - 1;
		}
	}

	for (int j = 0; j < _ny; j++)
	{
		auto ij0 = j * _nx;
		for (int i = 0; i < _nx; i++)
		{
			ijtol(i,j) = ijtol_[ij0 + i] - 1;
            rotflg(i,j) = rotflg_[ij0 + i];
		}
	}
	int nxyz6 = NEWSBT * _nxyz;
	_neib = new int[nxyz6];
	_hmesh = new GEOM_VAR[nxyz6];
	_lktosfc = new int[nxyz6];
	_vol = new GEOM_VAR[_nxyz];

	for (int k = 0; k < _nz; k++)
	{
		int l0 = k * _nxy;
		for (int l2d = 0; l2d < _nxy; l2d++)
		{
			int l = l0 + l2d;
			int lkd4 = l * NEWS;
			int lkd6 = l * NEWSBT;
			for (int inews = 0; inews < NEWS; inews++)
			{
				if (neibr(inews, l2d) <= -1) {
					neib(inews, l) = -1;
				} else {
					neib(inews, l) = l0 + neibr(inews, l2d);
				}
			}

			int lkb = (k - 1) * _nxy + l2d;
			int lkt = (k + 1) * _nxy + l2d;

			neib(BOT, l) = -1;
			neib(TOP, l) = -1;
			if (lkb > -1) neib(BOT,l) = lkb;
			if (lkt < _nxyz) neib(TOP,l) = lkt;

			for (int idir = 0; idir < NDIRMAX; idir++)
			{
				// in hemsh_, the zero value in 0th index.
				hmesh(idir, l) = hmesh_[l*NDIRMAX + idir];
			}
		}
	}

    for (int l = 0; l < _nxyz; ++l) {
        vol(l) = hmesh(XDIR,l)*hmesh(YDIR,l)*hmesh(ZDIR,l);
    }

	_hzcore = 0.0;

	for (int k = kbc(); k < kec(); k++)
	{
		int l = k * _nxy;
		_hzcore  += hmesh(ZDIR, l);
	}

	_idirlr = new int[_nsurf * LR];
	_sgnlr = new int[_nsurf * LR];
	_lklr = new int[_nsurf * LR];


	int ls = -1;

	for (int k = 0; k < _nz; k++)
	{
		int lk0 = k * _nxy;

		for (int j = 0; j < _ny; j++)
		{
			int ij0 =  j * _nx;
			++ls;
			idirlr(LEFT,ls) = YDIR;
			idirlr(RIGHT,ls) = XDIR;
			sgnlr(LEFT,ls) = MINUS;
			sgnlr(RIGHT,ls) = PLUS;
			int l = ijtol(_nxs[j],j);
			int lk = lk0 + l;
			lklr(LEFT, ls) = neib(WEST, lk);
			lklr(RIGHT, ls) = lk;

			for (int i = nxs(j); i < nye(j); i++)
			{
				int l = ijtol(i,j);

				int lk = lk0 + l;
				lktosfc(LEFT, XDIR, lk) = ls;
				lktosfc(RIGHT, XDIR, lk) = ++ls;
				lklr(LEFT, ls) = lk;
				lklr(RIGHT, ls) = neib(EAST, lk);
				idirlr(LEFT,ls) = XDIR;
				idirlr(RIGHT,ls) = XDIR;
				sgnlr(LEFT,ls) = PLUS;
				sgnlr(RIGHT,ls) = PLUS;
			}
		}

		for (int i = 0; i < _nx; i++)
		{
			int ij0 = i * _ny;
			++ls;
			int l = ijtol(i,nys(i));
			int lk = lk0 + l;

			idirlr(LEFT,ls) = XDIR;
			idirlr(RIGHT,ls) = YDIR;
			sgnlr(LEFT,ls) = MINUS;
			sgnlr(RIGHT,ls) = PLUS;
			lklr(LEFT, ls) = neib(NORTH, lk);
			lklr(RIGHT, ls) = lk;

			for (int j = nys(i); j < nye(i); j++)
			{
				int l = ijtol(i,j);

				int lk = lk0 + l;
				lktosfc(LEFT, YDIR, lk) = ls;
				lktosfc(RIGHT, YDIR, lk) = ++ls;
				lklr(LEFT, ls) = lk;
				lklr(RIGHT, ls) = neib(SOUTH, lk);
				idirlr(LEFT,ls) = YDIR;
				idirlr(RIGHT,ls) = YDIR;
				sgnlr(LEFT,ls) = PLUS;
				sgnlr(RIGHT,ls) = PLUS;
			}
		}
	}

	for (int l = 0; l < _nxy; l++)
	{
		++ls;
		idirlr(LEFT,ls) = ZDIR;
		idirlr(RIGHT,ls) = ZDIR;
		sgnlr(LEFT,ls) = PLUS;
		sgnlr(RIGHT,ls) = PLUS;

		int lk0 = l;
		lklr(LEFT, ls) = -1;
		for (int k = 0; k < _nz; k++)
		{
			int lk = k*_nxy+l;
			lktosfc(LEFT, ZDIR, lk) = ls;
			lklr(RIGHT, ls) = lk;

			lktosfc(RIGHT, ZDIR, lk) = ++ls;
			lklr(LEFT, ls) = lk;

			idirlr(LEFT,ls) = ZDIR;
			idirlr(RIGHT,ls) = ZDIR;
			sgnlr(LEFT,ls) = PLUS;
			sgnlr(RIGHT,ls) = PLUS;
		}
		lklr(RIGHT, ls) = -1;
	}
	_nsurf = ls+1;

	initAssemblyIndex();
	initCorner();
}

void Geometry::initAssemblyIndex()
{
	int nsub = (_part == 1 ? 0 : 1);

	_nya = (_ny + nsub) / 2;
	_nxa = (_nx + nsub) / 2;

	_nxsa = new int[_nya] {};
	_nxea = new int[_nya] {};
	_nysa = new int[_nxa] {};
	_nyea = new int[_nxa] {};

	_nxya = 0;
	for (int ja = 0; ja < _nya; ja++) {
		int j = ja * 2;
		nxsa(ja) = (nxs(j) + nsub) / 2;
		nxea(ja) = (nxe(j) + nsub) / 2;
		_nxya += nxea(ja) - nxsa(ja);
	}

	for (int ia = 0; ia < _nxa; ia++) {
		int i = ia * 2;
		nysa(ia) = (nys(i) + nsub) / 2;
		nyea(ia) = (nye(i) + nsub) / 2;
	}

	_ijtola = new int[_nxa * _nya];
	fill(_ijtola, _ijtola + _nxa * _nya, -1);

	_nxya = 0;
	for (int ja = 0; ja < _nya; ja++) {
		for (int ia = nxsa(ja); ia < nxea(ja); ia++) {
			ijtola(ia, ja) = _nxya++;
		}
	}

	_ltola = new int[_nxy] {};
    _latol = new int[_nxya*NEWS] {};
    _larot = new int[_nxya*NEWS] {};
	fill(_ltola, _ltola + _nxy, -1);
    fill(_latol, _latol + _nxya*NEWS, -1);
    fill(_larot, _latol + _nxya*NEWS, -1);

	_vola = new GEOM_VAR[_nxya*_nz]{};


	for (int j = 0; j < _ny; j++) {
		int ja = (j + nsub) / 2;
        int ji = (j + nsub) % 2;
		for (int i = nxs(j); i < nxe(j); i++){
			int ia = (i + nsub) / 2;
            int ii = (i + nsub) % 2;

			int l = ijtol(i, j);
			int la = ijtola(ia, ja);
			ltola(l) = la;
            int li = ji*2+ii;
            latol(li, la) = l;
            larot(li,la) = rotflg(i,j);
			for (int k = 0; k < _nz; k++)
			{
				vola(la + _nxya * k) += vol(l + _nxy * k);
			}
		}
	}

}

void Geometry::initCorner()
{
	_ncorn = nxe(0) - nxs(0) + 2;
	int ndown = _ncorn;
	for (int j = 1; j < ny(); j++) {
		int nupper = ndown;
		ndown = nxe(j) - nxs(j) + 2;
		_ncorn = _ncorn + max(nupper, ndown);
	}
	_ncorn = _ncorn + ndown;

	_ltolc = new int[4 * nxy()];
	_lctol = new int[4 * _ncorn];
	fill(_ltolc, _ltolc + 4 * nxy(), -1);
	fill(_lctol, _lctol + 4 * _ncorn, -1);


	int lc = 0;

	// the northest corner points
	ltolc(NW, ijtol(0, 0)) = lc;
	for (int i = nxs(0)+1; i < nxe(0); i++)
	{
		lc = lc + 1;
		ltolc(NE, ijtol(i - 1, 0)) = lc;
		ltolc(NW, ijtol(i, 0)) = lc;
	}
	lc = lc + 1;
	ltolc(NE, ijtol(nxe(0) - 1, 0)) = lc;

	for (int j = 1; j < ny(); j++)
	{
		if (nxs(j) > nxs(j - 1)) {
			for (int i = nxs(j-1); i < nxs(j); i++)
			{
				lc = lc + 1;
				if (i == nxs(j - 1)) {
					ltolc(SW, ijtol(i, j - 1)) = lc;
				}
				else {
					ltolc(SE, ijtol(i - 1, j - 1)) = lc;
					ltolc(SW, ijtol(i, j - 1)) = lc;
				}
			}
			ltolc(SE, ijtol(nxs(j) - 1, j - 1)) = lc + 1;
		}
		lc = lc + 1;
		ltolc(NW, ijtol(nxs(j), j)) = lc;

		for (int i = nxs(j)+1; i < nxe(j); i++)
		{
			lc = lc + 1;
			ltolc(NE, ijtol(i - 1, j)) = lc;
			ltolc(NW, ijtol(i, j)) = lc;

		}
		lc = lc + 1;
		ltolc(NE, ijtol(nxe(j)-1, j)) = lc;

		//adjusting orphaned corner point
		if (nxe(j) < nxe(j - 1)) {
			//when there are less nodes on the lower row.
			for (int i = nxe(j); i < nxe(j - 1); i++)
			{
				if (i == nxe(j)) {
					ltolc(SW, ijtol(i, j - 1)) = lc;
				}
				else {
					lc = lc + 1;
					ltolc(SE, ijtol(i - 1, j - 1)) = lc;
					ltolc(SW, ijtol(i, j - 1)) = lc;
				}
			}

			lc = lc + 1;
			ltolc(SE, ijtol(nxe(j - 1), j - 1)) = lc;
		}
	}

	//corners of the lowest row
	int je = ny() - 1;
	for (int i = nxs(je); i < nxe(je); i++)
	{
		lc = lc + 1;
		if (i == nxs(je)) {
			ltolc(SW, ijtol(i, je)) = lc;
		} 
		else {
			ltolc(SE, ijtol(i - 1, je)) = lc;
			ltolc(SW, ijtol(i, je)) = lc;
		}
	}
	lc = lc + 1;
	ltolc(SE, ijtol(nxe(je)-1, je)) = lc;

	for (int j = 0; j < ny()-1; j++)
	{
		int jp1 = j + 1;
		int ib = max(nxs(j), nxs(jp1));
		int ie = max(nxe(j), nxe(jp1));
		for (int i = ib; i < ie; i++)
		{
			ltolc(SW, ijtol(i, j)) = ltolc(NW, ijtol(i, jp1));
			ltolc(SE, ijtol(i, j)) = ltolc(NE, ijtol(i, jp1));
		}
	}
	if (_symang == 90) {
		int j = 0;
		int i = 0;
		for (int inews = 0; inews < NEWS; inews++)
		{
			ltolc(inews, ijtol(i, j)) = ltolc(SE, ijtol(i, j));
		}
		for (int i = nxs(j)+1; i < nxe(j); i++)
		{
			ltolc(NW, ijtol(i, j)) = ltolc(NE, ijtol(j, i));
			ltolc(NE, ijtol(i, j)) = ltolc(SE, ijtol(j, i));
			ltolc(NW, ijtol(j, i)) = ltolc(SW, ijtol(i, j));
			ltolc(SW, ijtol(j, i)) = ltolc(SE, ijtol(i, j));
		}
	}

	//for (int l = 0; l < nxy(); l++)
	//{
	//	lctol(NW, ltolc(SE, l)) = l;
	//	lctol(NE, ltolc(SW, l)) = l;
	//	lctol(SE, ltolc(NW, l)) = l;
	//	lctol(SW, ltolc(NE, l)) = l;
	//}
	

}
