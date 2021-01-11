#pragma once
#include "pch.h"

/**
 * Nodal Geometry
 * The naming rule for indexing variables
 *
 * the number of nodes : n+(x,y)+(a)+(f)
 * n    : number
 * x    : x-direction
 * y    : y-direction
 * xy   : 2D(xy) plane
 * a    : asembly-wise
 * f    :: fuel only
 *
 * one-dimensional indices : (i,j,k)+(s,e)+(a)+(f)
 * i    : x-direction
 * j    : y-direction,
 * k    : z-direction
 * s    : starting
 * e    : ending
 * a    : assembly-wise
 * f    : fuel only
 *
 * note that indices follows the c-style index numbering
 * starting index   : 0     (included)
 * ending index     : n+1   (not included)
 *
 * Two-dimensional indices
 * la   : assembly index in 2D
 * l    : node index in 2D
 *
 */





class Geometry
{
private:
	int _ng;
	int	_nxy;
	int	_nz;
	int	_nxyz;

	int _nx;
	int _ny; 
	int* _nxs; 
	int* _nxe;
	int* _nys;
	int* _nye;
	int		_nsurf;
	int*	_neibr;
	int*	_ijtol;
	int*	_neib;

	int*	_lklr;
	int*	_idirlr;
	int*	_sgnlr;
	int*	_lktosfc;

	float* _hmesh;


public:
	Geometry();
	virtual ~Geometry();

	void init(int* ng, int* nxy, int* nz, int* nx, int* ny, int* nxs, int* nxe, int* nys, int* nye, int* nsurf_, int* ijtol, int* neibr, double* hmesh);

	inline int& ng() { return _ng; };
	inline int& nxy() { return _nxy; };
	inline int& nz() { return _nz; };
	inline int& nxyz() { return _nxyz; };
	inline int& nsurf() { return _nsurf; };
	inline int& nx() { return _nx; };
	inline int& ny() { return _ny; };

	inline int& nxs(const int& j) { return _nxs[j]; };
	inline int& nxe(const int& j) { return _nxe[j]; };
	inline int& nys(const int& i) { return _nys[i]; };
	inline int& nye(const int& i) { return _nye[i]; };
	inline int& neibr(const int& news, const int& l) { return _neibr[l*NEWS + news]; };
	inline int& ijtol(const int& i, const int& j) { return _ijtol[j*_nx + i]; };
	inline int& neib(const int& newsbt, const int& lk) { return _neib[lk * NEWSBT + newsbt]; };
	inline int& lklr(const int& lr, const int& ls) { return _lklr[ls * LR + lr]; };
	inline int& idirlr(const int& lr, const int& ls) { return _idirlr[ls * LR + lr]; };
	inline int& sgnlr(const int& lr, const int& ls) { return _sgnlr[ls * LR + lr]; };
	inline int& lktosfc(const int& lr, const int& idir, const int& lk) { return _lktosfc[(lk * NDIRMAX + idir)*LR + lr]; };

	inline float& hmesh(const int& idir, const int& lk) { return _hmesh[lk * NDIRMAX + idir]; };
};

