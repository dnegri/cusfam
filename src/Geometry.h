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
protected:
    int _ndivxy;
    int _ndivxy2;
    int _ncellxy;
	int _ng;
	int _ng2;
	int	_nxy;
	int	_nz;
	int	_nxyz;
	int	_ngxy;
	int	_ngxyz;

	int _kbc;
	int _kec;

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
	int* _ltola;
    int* _latol;
    int* _larot;

	int _nxyfa;
	int _ncellfa;


	int _nxa;
	int _nya;
	int* _nxsa;
	int* _nxea;
	int* _nysa;
	int* _nyea;
	int _nxya;
	int* _ijtola;


	int* _comps;
	int  _ncomp;
	char _compnames[50][13];
	int* _hffs;
	int  _nhff;
	char _hffnames[50][13];

	int* _lklr;
	int* _idirlr;
	int* _sgnlr;
	int* _lktosfc;

	int _ncorn;
	int* _ltolc;
	int* _lctol;

	int _symopt;
	int _symang;
	double* _albedo;

	double* _hmesh;
	double* _hz;
	double* _vol;
	double* _vola;

	double _part;
	double _hzcore;

public:
	Geometry();
	virtual ~Geometry();

	void setBoundaryCondition(int symopt, int symang, float* albedo);
	void initDimension(int ng_, int nxy_, int nz_, int nx_, int ny_, int nsurf_, int ndivxy_);
	void initIndex(int* nxs, int* nxe, int* nys, int* nye, int * ijtol_, int* neibr_, float* hmesh_);
	void initAssemblyIndex(int nxyfa, int ncellfa, int * latol_, int * larot_);
	void initCorner(const int& ncorn, const int* lctol, const int* ltolc);

    inline int& ndivxy() { return _ndivxy; };
    inline int& ncellxy() { return _ncellxy; };
	inline int& ncellfa() { return _ncellfa; };
	inline int& nxyfa() { return _nxyfa; };
	
	inline int& ng() { return _ng; };
	inline int& ng2() { return _ng2; };
	inline int& nxy() { return _nxy; };
	inline int& nz() { return _nz; };
	inline int& nxyz() { return _nxyz; };
	inline int& ngxyz() { return _ngxyz; };
	inline int& ngxy() { return _ngxy; };
	inline int& nsurf() { return _nsurf; };
	inline int& nx() { return _nx; };
	inline int& ny() { return _ny; };
	inline int& symopt() { return _symopt; };
	inline int& symang() { return _symang; };
	inline int& kbc() { return _kbc; };
	inline int& kec() { return _kec; };
	inline double& part() { return _part; };

	inline const int& ng() const { return _ng; };
	inline const int& ng2() const { return _ng2; };
	inline const int& nxy() const { return _nxy; };
	inline const int& nz() const { return _nz; };
	inline const int& nxyz() const { return _nxyz; };
	inline const int& ngxyz() const { return _ngxyz; };
	inline const int& ngxy() const { return _ngxy; };
	inline const int& nsurf() const { return _nsurf; };
	inline const int& nx() const { return _nx; };
	inline const int& ny() const { return _ny; };
	inline const int& symopt() const { return _symopt; };
	inline const int& symang() const { return _symang; };

	inline const int& ncorn() const { return _ncorn; };
	inline int* ltolc() const { return _ltolc; }
	inline int* lctol() const { return _lctol; }
	inline int& ltolc(const int& news, const int& l) const { return _ltolc[l*NEWS+ news]; }
	inline int& lctol(const int& news, const int& lc) const { return _lctol[lc * NEWS + news]; }

	const double* albedo() const { return _albedo; }
	const int* neibr() const { return _neibr; }
	const int* ijtol() const { return _ijtol; }
	const int* nxs() const { return _nxs; }
	const int* nxe() const { return _nxe; }
	const int* nys() const { return _nys; }
	const int* nye() const { return _nye; }
	const int* neib() const { return _neib; }
	const double* hmesh() const { return _hmesh; }
	const int* lktosfc() const { return _lktosfc; }
	const double* vol() const { return _vol; }
	const int* idirlr() const { return _idirlr; }
	const int* sgnlr() const { return _sgnlr; }
	const int* lklr() const { return _lklr; }

	inline int& nxs(const int& j) { return _nxs[j]; };
	inline int& nxe(const int& j) { return _nxe[j]; };
	inline int& nys(const int& i) { return _nys[i]; };
	inline int& nye(const int& i) { return _nye[i]; };
	inline int& neibr(const int& news, const int& l) { return _neibr[l * NEWS + news]; };
	inline int& ijtol(const int& i, const int& j) { return _ijtol[j * _nx + i]; };
	inline int& neib(const int& newsbt, const int& lk) { return _neib[lk * NEWSBT + newsbt]; };
	inline int& neib(const int& lr, const int& idir, const int& lk) { return _neib[lk * NEWSBT + idir * LR + lr]; };
	inline int& lklr(const int& lr, const int& ls) { return _lklr[ls * LR + lr]; };
	inline int& idirlr(const int& lr, const int& ls) { return _idirlr[ls * LR + lr]; };
	inline int& sgnlr(const int& lr, const int& ls) { return _sgnlr[ls * LR + lr]; };
	inline int& lktosfc(const int& lr, const int& idir, const int& lk) { return _lktosfc[(lk * NDIRMAX + idir) * LR + lr]; };

	inline const int& nxya() const { return _nxya; };
	inline const int& nya() const { return _nya; };
	inline const int& nxa() const { return _nxa; };
	inline int& nxsa(const int& ja) { return _nxsa[ja]; };
	inline int& nxea(const int& ja) { return _nxea[ja]; };
	inline int& nysa(const int& ia) { return _nysa[ia]; };
	inline int& nyea(const int& ia) { return _nyea[ia]; };
	inline int& ijtola(const int& ia, const int& ja) { return _ijtola[ja * _nxa + ia]; };
	inline int& ltola(const int& l) { return _ltola[l]; };
    inline int& latol(const int& li, const int& la) { return _latol[la*NEWS+li]; };
    inline int& larot(const int& li, const int& la) { return _larot[la*NEWS+li]; };
    inline int* larot(const int& la) { return &_larot[la*NEWS]; };

	inline int& comp(const int& lk) { return _comps[lk]; };
	inline int* comps() { return _comps; };
	inline int& ncomp() { return _ncomp; };
	char** compnames() { return (char**)_compnames; };

	inline int& hff(const int& lk) { return _hffs[lk]; };
	inline int* hffs() { return _hffs; };
	inline int& nhff() { return _nhff; };
	char** hffnames() { return (char**)_hffnames; };

	inline double& hmesh(const int& idir, const int& lk) { return _hmesh[lk * NDIRMAX + idir]; };
	inline double& hz(const int& k) { return _hz[k]; };
	inline double& vol(const int& lk) { return _vol[lk]; };
	inline double& vola(const int& lka) { return _vola[lka]; };
	inline double& albedo(const int& lr, const int& idir) { return _albedo[idir * LR + lr]; };
	inline double& hzcore() { return _hzcore; };

};

