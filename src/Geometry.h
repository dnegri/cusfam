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





class Geometry : public Managed {
protected :
	int _ng;
	int _ng2;
	int	_nxy;
	int	_nz;
	int	_nxyz;
    int	_ngxy;
    int	_ngxyz;

	int _nx;
	int _ny; 
	int* _nxs; 
	int* _nxe;
	int* _nys;
	int* _nye;
	int	 _nsurf;
	int* _neibr;
	int* _ijtol;
	int* _neib;

	int* _comps;
	int  _ncomp;
	char _compnames[50][13];

	int* _lklr;
	int* _idirlr;
	int* _sgnlr;
	int* _lktosfc;

	int _symopt;
	int _symang;
	GEOM_VAR* _albedo;

	GEOM_VAR* _hmesh;
	GEOM_VAR* _vol;
	
	GEOM_VAR _part;

public:
	__host__ Geometry();
	__host__ virtual ~Geometry();

	__host__ __device__ void setBoundaryCondition(int* symopt, int* symang, float* albedo);
	__host__ __device__ void initDimension(int* ng_, int* nxy_, int* nz_, int* nx_, int* ny_, int* nsurf_);
    __host__ __device__ void initIndex(int* nxs, int* nxe, int* nys, int* nye, int* ijtol, int* neibr, float* hmesh);

	__host__ __device__ inline int& ng() { return _ng; };
	__host__ __device__ inline int& ng2() { return _ng2; };
	__host__ __device__ inline int& nxy() { return _nxy; };
	__host__ __device__ inline int& nz() { return _nz; };
	__host__ __device__ inline int& nxyz() { return _nxyz; };
    __host__ __device__ inline int& ngxyz() { return _ngxyz; };
    __host__ __device__ inline int& ngxy() { return _ngxy; };
	__host__ __device__ inline int& nsurf() { return _nsurf; };
	__host__ __device__ inline int& nx() { return _nx; };
	__host__ __device__ inline int& ny() { return _ny; };
	__host__ __device__ inline int& symopt() { return _symopt; };
	__host__ __device__ inline int& symang() { return _symang; };
	__host__ __device__ inline GEOM_VAR& part() { return _part; };

	__host__ __device__ inline const int& ng() const { return _ng; };
	__host__ __device__ inline const int& ng2() const { return _ng2; };
	__host__ __device__ inline const int& nxy() const { return _nxy; };
	__host__ __device__ inline const int& nz() const { return _nz; };
	__host__ __device__ inline const int& nxyz() const { return _nxyz; };
	__host__ __device__ inline const int& ngxyz() const { return _ngxyz; };
	__host__ __device__ inline const int& ngxy() const { return _ngxy; };
	__host__ __device__ inline const int& nsurf() const { return _nsurf; };
	__host__ __device__ inline const int& nx() const { return _nx; };
	__host__ __device__ inline const int& ny() const { return _ny; };
	__host__ __device__ inline const int& symopt() const { return _symopt; };
	__host__ __device__ inline const int& symang() const { return _symang; };

	__host__ __device__ const GEOM_VAR* albedo() const { return _albedo; }
	__host__ __device__ const int* neibr() const { return _neibr; }
	__host__ __device__ const int* ijtol() const { return _ijtol; }
	__host__ __device__ const int* nxs() const { return _nxs; }
	__host__ __device__ const int* nxe() const { return _nxe; }
	__host__ __device__ const int* nys() const { return _nys; }
	__host__ __device__ const int* nye() const { return _nye; }
	__host__ __device__ const int* neib() const { return _neib; }
	__host__ __device__ const GEOM_VAR* hmesh() const { return _hmesh; }
	__host__ __device__ const int* lktosfc() const { return _lktosfc; }
	__host__ __device__ const GEOM_VAR* vol() const { return _vol; }
	__host__ __device__ const int* idirlr() const { return _idirlr; }
	__host__ __device__ const int* sgnlr() const { return _sgnlr; }
	__host__ __device__ const int* lklr() const { return _lklr; }

	__host__ __device__ inline int& nxs(const int& j) { return _nxs[j]; };
	__host__ __device__ inline int& nxe(const int& j) { return _nxe[j]; };
	__host__ __device__ inline int& nys(const int& i) { return _nys[i]; };
	__host__ __device__ inline int& nye(const int& i) { return _nye[i]; };
	__host__ __device__ inline int& neibr(const int& news, const int& l) { return _neibr[l*NEWS + news]; };
	__host__ __device__ inline int& ijtol(const int& i, const int& j) { return _ijtol[j*_nx + i]; };
	__host__ __device__ inline int& neib(const int& newsbt, const int& lk) { return _neib[lk * NEWSBT + newsbt]; };
    __host__ __device__ inline int& neib(const int& lr, const int& idir, const int& lk) { return _neib[lk * NEWSBT + idir*LR + lr]; };
	__host__ __device__ inline int& lklr(const int& lr, const int& ls) { return _lklr[ls * LR + lr]; };
	__host__ __device__ inline int& idirlr(const int& lr, const int& ls) { return _idirlr[ls * LR + lr]; };
	__host__ __device__ inline int& sgnlr(const int& lr, const int& ls) { return _sgnlr[ls * LR + lr]; };
	__host__ __device__ inline int& lktosfc(const int& lr, const int& idir, const int& lk) { return _lktosfc[(lk * NDIRMAX + idir)*LR + lr]; };

	__host__ __device__ inline int& comp(const int& l) { return _comps[l]; };
	__host__ __device__ inline int* comp() { return _comps; };
	__host__ __device__ inline int& ncomp() { return _ncomp; };
	__host__ __device__ char** compnames() { return (char**)_compnames; };

	__host__ __device__ inline GEOM_VAR& hmesh(const int& idir, const int& lk) { return _hmesh[lk * NDIRMAX + idir]; };
    __host__ __device__ inline GEOM_VAR& vol(const int& lk) { return _vol[lk]; };
	__host__ __device__ inline GEOM_VAR& albedo(const int& lr, const int& idir) { return _albedo[idir * LR + lr]; };

};

