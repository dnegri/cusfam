#include "Geometry.h"

Geometry::Geometry()
{
}

Geometry::~Geometry()
{
}

void Geometry::setBoundaryCondition(int symopt, int symang, float* albedo)
{
	_symopt = symopt;
	_symang = symang;
	_albedo = new double[LR*NDIRMAX];
	for (int idir = 0; idir < NDIRMAX; idir++)
	{
		for (int l = 0; l < LR; l++)
		{
			_albedo[idir * LR + l] = albedo[idir * LR + l];
		}
	}

	_part = _symang / 360.0;
}

void Geometry::initDimension(int ng_, int nxy_, int nz_, int nx_, int ny_, int nsurf_, int ndivxy_) {
    _ndivxy = ndivxy_;
	_ng = ng_;
	_nx = nx_;
	_ny = ny_;
	_nz = nz_;
	_nxy = nxy_;
	_nxyz = _nxy * _nz;
	_nsurf = nsurf_;
	_ng2 = _ng * _ng;
	_ngxyz = _nxyz * _ng;
	_ngxy = _nxy * _ng;
	_kbc = 1;
	_kec = _nz - 1;

	_ncellxy = 16;
	_ncellfa = 236;
    _ndivxy2 = ndivxy_*ndivxy_;

}

void Geometry::initIndex(int* nxs_, int* nxe_, int* nys_, int* nye_, int * ijtol_, int* neibr_, float* hmesh_)
{
	_neibr = new int[_nxy*NEWS];
	_ijtol = new int[_nx*_ny];
	_nxs = new int[_ny];
	_nxe = new int[_ny];
	_nys = new int[_nx];
	_nye = new int[_nx];

	_comps = new int[_nxyz];
	_hffs = new int[_nxyz];

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
		}
	}
	int nxyz6 = NEWSBT * _nxyz;
	_neib = new int[nxyz6];
	_hmesh = new double[NDIRMAX*_nxyz];
	_lktosfc = new int[nxyz6];
	_vol = new double[_nxyz];

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

	_hz = new double[_nz];
	for (int k = 0; k < _nz; k++)
	{
		int l = k * _nxy;
		hz(k) = hmesh(ZDIR, l);
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
}

void Geometry::initAssemblyIndex(int nxyfa_, int ncellfa_, int * latol_, int * larot_)
{
	_nxyfa = nxyfa_;
	_ncellfa = ncellfa_;

	int nsub = (_part == 1 || _ndivxy == 1 ? 0 : 1);

	_nya = (_ny + nsub) / _ndivxy;
	_nxa = (_nx + nsub) / _ndivxy;

	_nxsa = new int[_nya] {};
	_nxea = new int[_nya] {};
	_nysa = new int[_nxa] {};
	_nyea = new int[_nxa] {};

	_nxya = 0;
	for (int ja = 0; ja < _nya; ja++) {
		int j = ja * 2;
		nxsa(ja) = (nxs(j) + nsub) / _ndivxy;
		nxea(ja) = (nxe(j) + nsub) / _ndivxy;
		_nxya += nxea(ja) - nxsa(ja);
	}

	for (int ia = 0; ia < _nxa; ia++) {
		int i = ia * 2;
		nysa(ia) = (nys(i) + nsub) / _ndivxy;
		nyea(ia) = (nye(i) + nsub) / _ndivxy;
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
    _latol = new int[_nxya*_ndivxy2] {};
    _larot = new int[_nxya*_ndivxy2] {};
	fill(_ltola, _ltola + _nxy, -1);
    copy(latol_, latol_ + _nxya*_ndivxy2, _latol);
    copy(larot_, larot_ + _nxya*_ndivxy2, _larot);

	for (int la = 0; la < _nxya; la++)
	{
		for (int li = 0; li < NEWS; li++)
		{
			latol(li, la)--;
		}
	}


	_vola = new double[_nxya*_nz]{};

	for (int j = 0; j < _ny; j++) {
		int ja = (j + nsub) / _ndivxy;
        int ji = (j + nsub) % _ndivxy;
		for (int i = nxs(j); i < nxe(j); i++){
			int ia = (i + nsub) / _ndivxy;
            int ii = (i + nsub) % _ndivxy;

			int l = ijtol(i, j);
			int la = ijtola(ia, ja);
			ltola(l) = la;
			for (int k = 0; k < _nz; k++)
			{
				vola(la + _nxya * k) += vol(l + _nxy * k);
			}
		}
	}

}

void Geometry::initCorner(const int& ncorn_, const int* lctol_, const int* ltolc_)
{
	_ncorn = ncorn_;

	_lctol = new int[NEWS * ncorn()];
	_ltolc = new int[NEWS * nxy()];

	copy(lctol_, lctol_ + NEWS * ncorn(), _lctol);
	copy(ltolc_, ltolc_ + NEWS * nxy(), _ltolc);

	for (int i = 0; i < NEWS*ncorn(); i++)
	{
		--_lctol[i];
	}

	for (int i = 0; i < NEWS*nxy(); i++)
	{
		--_ltolc[i];
	}

}
