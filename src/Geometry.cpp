#include "Geometry.h"

Geometry::Geometry()
{
}

Geometry::~Geometry()
{
}

void Geometry::init(int* ng_, int* nxy_, int* nz_, int* nx_, int* ny_, int* nxs_, int* nxe_, int* nys_, int* nye_, int* nsurf_, int * ijtol_, int* neibr_, double* hmesh_)
{
	_ng = *ng_;
	_nx = *nx_;
	_ny = *ny_;
	_nz = *nz_;
	_nxy = *nxy_;
	_nxyz = _nxy * _nz;
	_nsurf = *nsurf_ * _nz + (_nz + 1) * _nxy;

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
				neib(inews, lk) = lk0 + neibr(inews, l);
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
				hmesh(idir, lk) = hmesh_[lkd4 + idir + 1];
			}
		}
	}



	_idirlr = new int[_nsurf * LR];
	_sgnlr = new int[_nsurf * LR];
	_lklr = new int[_nsurf * LR];


	int is = -1;

	for (size_t k = 0; k < _nz; k++)
	{
		int lk0 = k * _nxy;

		for (size_t j = 0; j < _ny; j++)
		{
			int ij0 =  j * _nx;
			++is;
			idirlr(LEFT,is) = YDIR;
			idirlr(RIGHT,is) = XDIR;
			sgnlr(LEFT,is) = MINUS;
			sgnlr(RIGHT,is) = PLUS;
			int l = ijtol(_nxs[j],j);
			int lk = lk0 + l;
			lklr(LEFT, is) = neib(WEST, lk);
			lklr(RIGHT, is) = lk;

			for (size_t i = nxs(j); i < nye(j); i++)
			{
				int l = ijtol(i,j);

				int lk = lk0 + l;
				lktosfc(LEFT, XDIR, lk) = is;
				lktosfc(RIGHT, XDIR, lk) = ++is;
				lklr(LEFT, is) = lk;
				lklr(RIGHT, is) = neib(EAST, lk);
				idirlr(LEFT,is) = XDIR;
				idirlr(RIGHT,is) = XDIR;
				sgnlr(LEFT,is) = PLUS;
				sgnlr(RIGHT,is) = PLUS;
			}
		}

		for (size_t i = 0; i < _nx; i++)
		{
			int ij0 = i * _ny;
			++is;
			int l = ijtol(i,nys(i));
			int lk = lk0 + l;

			idirlr(LEFT,is) = XDIR;
			idirlr(RIGHT,is) = YDIR;
			sgnlr(LEFT,is) = MINUS;
			sgnlr(RIGHT,is) = PLUS;
			lklr(LEFT, is) = neib(NORTH, lk);
			lklr(RIGHT, is) = lk;

			for (size_t j = nys(i); j < nye(i); j++)
			{
				int l = ijtol(i,j);

				int lk = lk0 + l;
				lktosfc(LEFT, YDIR, lk) = is;
				lktosfc(RIGHT, YDIR, lk) = ++is;
				lklr(LEFT, is) = lk;
				lklr(RIGHT, is) = neib(SOUTH, lk);
				idirlr(LEFT,is) = YDIR;
				idirlr(RIGHT,is) = YDIR;
				sgnlr(LEFT,is) = PLUS;
				sgnlr(RIGHT,is) = PLUS;
			}
		}
	}

	for (size_t l = 0; l < _nxy; l++)
	{
		++is;
		idirlr(LEFT,is) = ZDIR;
		idirlr(RIGHT,is) = ZDIR;
		sgnlr(LEFT,is) = PLUS;
		sgnlr(RIGHT,is) = PLUS;

		int lk0 = l;
		lklr(LEFT, is) = -1;
		for (size_t k = 0; k < _nz; k++)
		{
			int lk = k*_nxy+l;
			lktosfc(LEFT, ZDIR, lk) = is;
			lklr(RIGHT, is) = lk;

			lktosfc(RIGHT, ZDIR, lk) = ++is;
			lklr(LEFT, is) = lk;

			idirlr(LEFT,is) = ZDIR;
			idirlr(RIGHT,is) = ZDIR;
			sgnlr(LEFT,is) = PLUS;
			sgnlr(RIGHT,is) = PLUS;
		}
		lklr(LEFT, is) = -1;
	}
	_nsurf = is+1;
}
