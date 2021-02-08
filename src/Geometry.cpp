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
	_albedo = new float[LR*NDIRMAX];
	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		for (size_t l = 0; l < LR; l++)
		{
			_albedo[idir * LR + l] = albedo[idir * LR + l];
		}
	}
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
}

void Geometry::initIndex(int* nxs_, int* nxe_, int* nys_, int* nye_, int * ijtol_, int* neibr_, float* hmesh_)
{
	_neibr = new int[_nxy*NEWS];
	_ijtol = new int[_nx*_ny];
	_nxs = new int[_ny];
	_nxe = new int[_ny];
	_nys = new int[_nx];
	_nye = new int[_nx];

	for (size_t j = 0; j < _ny; j++)
	{
		nxs(j) = nxs_[j]-1;
		nxe(j) = nxe_[j];
	}
	for (size_t i = 0; i < _nx; i++)
	{
		nys(i) = nys_[i] - 1;
		nye(i) = nye_[i];
	}

	for (size_t l = 0; l < _nxy; l++)
	{
		auto lnews0 = l * NEWS;
		for (size_t i4 = 0; i4 < NEWS; i4++)
		{
			neibr(i4,l) = neibr_[lnews0 + i4] - 1;
		}
	}

	for (size_t j = 0; j < _ny; j++)
	{
		auto ij0 = j * _nx;
		for (size_t i = 0; i < _nx; i++)
		{
			ijtol(i,j) = ijtol_[ij0 + i] - 1;
		}
	}
	int nxyz6 = NEWSBT * _nxyz;
	_neib = new int[nxyz6];
	_hmesh = new float[nxyz6];
	_lktosfc = new int[nxyz6];
	_vol = new float[_nxyz];

	for (size_t k = 0; k < _nz; k++)
	{
		int lk0 = k * _nxy;
		for (size_t l = 0; l < _nxy; l++)
		{
			int lk = lk0 + l;
			int lkd4 = lk * NEWS;
			int lkd6 = lk * NEWSBT;
			for (size_t inews = 0; inews < NEWS; inews++)
			{
				if (neibr(inews, l) <= -1) {
					neib(inews, lk) = -1;
				} else {
					neib(inews, lk) = lk0 + neibr(inews, l);
				}
			}

			int lkb = (k - 1) * _nxy + l;
			int lkt = (k + 1) * _nxy + l;

			neib(BOT, lk) = -1;
			neib(TOP, lk) = -1;
			if (lkb > -1) neib(BOT,lk) = lkb;
			if (lkt < _nxyz) neib(TOP,lk) = lkt;

			for (size_t idir = 0; idir < NDIRMAX; idir++)
			{
				// in hemsh_, the zero value in 0th index.
				hmesh(idir, lk) = hmesh_[lk*NDIRMAX + idir];
			}
		}
	}

    for (int l = 0; l < _nxyz; ++l) {
        vol(l) = hmesh(XDIR,l)*hmesh(YDIR,l)*hmesh(ZDIR,l);
    }



	_idirlr = new int[_nsurf * LR];
	_sgnlr = new int[_nsurf * LR];
	_lklr = new int[_nsurf * LR];


	int ls = -1;

	for (size_t k = 0; k < _nz; k++)
	{
		int lk0 = k * _nxy;

		for (size_t j = 0; j < _ny; j++)
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

			for (size_t i = nxs(j); i < nye(j); i++)
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

		for (size_t i = 0; i < _nx; i++)
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

			for (size_t j = nys(i); j < nye(i); j++)
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

	for (size_t l = 0; l < _nxy; l++)
	{
		++ls;
		idirlr(LEFT,ls) = ZDIR;
		idirlr(RIGHT,ls) = ZDIR;
		sgnlr(LEFT,ls) = PLUS;
		sgnlr(RIGHT,ls) = PLUS;

		int lk0 = l;
		lklr(LEFT, ls) = -1;
		for (size_t k = 0; k < _nz; k++)
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
