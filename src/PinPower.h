#pragma once
#include "pch.h"
#include "Geometry.h"
#include "CrossSection.h"

class PinPower
{
private:

	Geometry& _g;
    CrossSection& _x;

	bool usemss = true;
	bool term15 = true;
	int  nmaxppr = 20;
    int  nterm = 15;

	int _ncell_plane;

	double  sqrt2 = 1.41421356;
    double  rsqrt2 = 0.70710678;

	int        npin;

	double  * _kappa,
	        * _qf2d,
			* _qc2d,
			* _pc2d,
			* _hc2d,
			* _jcornx,
			* _jcorny;

	//   coefficient of least square fitting method
	double  * _clsqf01,
			* _clsqf02,
			* _clsqf11,
			* _clsqf12,
			* _clsqf21,
			* _clsqf22,
			* _clsqf31,
			* _clsqf32,
			* _clsqf41,
			* _clsqf42,
			* _clsqfx1y1,
			* _clsqf1221,
			* _clsqf1331,
			* _clsqfx2y2;

	//   coefficient of general solutions
	double  * _cpc02,
			* _cpc04,
			* _cpc022,
			* _cpc11,
			* _cpc12,
			* _cpc21,
			* _cpc22,
			* _chc6,
			* _chc13j,
			* _chc13p,
			* _chc57j,
			* _chc57p,
			* _chc8j,
			* _chc8p,
			* _chc24j,
			* _chc24a;

	//   coefficiets of corner partial current
	double  * _cpjxh1,
			* _cpjxh2,
			* _cpjxh5,
			* _cpjxh6,
			* _cpjxh7,
			* _cpjxh8,
			* _cpjxp6,
			* _cpjxp7,
			* _cpjxp8,
			* _cpjxp9,
			* _cpjxp11,
			* _cpjxp12,
			* _cpjxp13,
			* _cpjxp14,
			* _cpjyh3,
			* _cpjyh4,
			* _cpjyh5,
			* _cpjyh6,
			* _cpjyh7,
			* _cpjyh8,
			* _cpjyp2,
			* _cpjyp3,
			* _cpjyp4,
			* _cpjyp9,
			* _cpjyp10,
			* _cpjyp12,
			* _cpjyp13,
			* _cpjyp14;

	double  * _phicorn,
			* _phicorn0,
			* _avgjnetz;     //(ng,nxy,nz)

	double* _trlzcff;     //(9,ng,nxy,nz)


    int     _nrest,
            _npex,
            _npinxy;

    double  * _hpini,
            * _hpinj,
            * _pcoeff;

    double  * _pinpowa,
            * _pinphia;     //(ng,nxy,nz)

    double* _pow3da;
    double* _peaka;

    int _fxyb=6, _fxyt=22;
public:
	PinPower(Geometry& g, CrossSection& x);
	virtual ~PinPower();


	inline double& phicorn(const int& ig, const int& lc, const int& k) {
		return _phicorn[k*_g.ncorn()*_g.ng()+lc*_g.ng()+ig];
	}
    inline double& qf2d(const int& order, const int& ig, const int& lk) {
        return _qf2d[lk*_g.ng()*nterm+ig*nterm+order];
    }
    inline double& qc2d(const int& order, const int& ig, const int& lk) {
        return _qc2d[lk*_g.ng()*nterm+ig*nterm+order];
    }
    inline double& pc2d(const int& order, const int& ig, const int& lk) {
        return _pc2d[lk*_g.ng()*nterm+ig*nterm+order];
    }
    inline double& hc2d(const int& order, const int& ig, const int& lk) {
        return _hc2d[lk*_g.ng()*8+ig*8+order];
    }
    inline double& cpc02(const int& ig, const int& lk) {
        return _cpc02[lk*_g.ng()+ig];
    }
    inline double& cpc04(const int& ig, const int& lk) {
        return _cpc04[lk*_g.ng()+ig];
    }
    inline double& cpc022(const int& ig, const int& lk) {
        return _cpc022[lk*_g.ng()+ig];
    }
    inline double& cpc11(const int& ig, const int& lk) {
        return _cpc11[lk*_g.ng()+ig];
    }
    inline double& cpc12(const int& ig, const int& lk) {
        return _cpc12[lk*_g.ng()+ig];
    }
    inline double& cpc21(const int& ig, const int& lk) {
        return _cpc21[lk*_g.ng()+ig];
    }
    inline double& cpc22(const int& ig, const int& lk) {
        return _cpc22[lk*_g.ng()+ig];
    }
    inline double& chc6(const int& ig, const int& lk) {
        return _chc6[lk*_g.ng()+ig];
    }
    inline double& chc13j(const int& ig, const int& lk) {
        return _chc13j[lk*_g.ng()+ig];
    }
    inline double& chc13p(const int& ig, const int& lk) {
        return _chc13p[lk*_g.ng()+ig];
    }
    inline double& chc57j(const int& ig, const int& lk) {
        return _chc57j[lk*_g.ng()+ig];
    }
    inline double& chc57p(const int& ig, const int& lk) {
        return _chc57p[lk*_g.ng()+ig];
    }
    inline double& chc8j(const int& ig, const int& lk) {
        return _chc8j[lk*_g.ng()+ig];
    }
    inline double& chc8p(const int& ig, const int& lk) {
        return _chc8p[lk*_g.ng()+ig];
    }
    inline double& chc24j(const int& ig, const int& lk) {
        return _chc24j[lk*_g.ng()+ig];
    }
    inline double& chc24a(const int& ig, const int& lk) {
        return _chc24a[lk*_g.ng()+ig];
    }

    inline double& trlzcff(const int& yorder, const int& xorder, const int& ig, const int& lk) {
        return _trlzcff[lk*_g.ng()*9+ig*9+xorder*3 + yorder];
    }

    inline double& kappa(const int& ig, const int& lk) {
        return _kappa[lk*_g.ng()+ig];
    }

    inline double& hpinj(const int& jpinxy, const int& li) {
        return _hpinj[li*_npinxy+jpinxy];
    }

    inline double& hpini(const int& ipinxy, const int& li) {
        return _hpinj[li*_npinxy+ipinxy];
    }
    inline double& pcoeff(const int& order, const int& ipinxy, const int& jpinxy, const int& li) {
        return _pcoeff[(li*_npinxy*_npinxy+jpinxy*_npinxy+ipinxy)*nterm + order];
    }


    inline double& pinpowa(const int& ipa, const int& jpa, const int& la, const int& k) {
        return _pinpowa[((k*_g.nxya()+la)*_g.ncellxy()*_g.ncellxy()+jpa*_g.ncellxy()+ipa)];
    }

    inline double& pinphia(const int& ig, const int& ipa, const int& jpa, const int& la, const int& k) {
        return _pinphia[((k*_g.nxy()+la)*_g.ncellxy()*_g.ncellxy()+jpa*_g.ncellxy()+ipa)*_g.ng()+ig];
    }

	void calhomo(const double& eigv, double* flux, double* phis, double* jnet);
	void calpinpower();

	void calphicorn(double* flux, double* phis);
	void caltrlz(int l, int k, double* jnet);
    void calcff(int l, int k);
    void calsol(int l, int k, double* jnet);
    void calsol2drhs(int l, int k, const double & reigv );
	void calpinpower(const int& la, const int& k, const int* larot1a);
	void applyFF(void* ff_ptr, float * burn);
    void printPinPower(int k);
    double& pow3da(int la, int k) {return _pow3da[k * _g.nxya() + la]; };
    double& peaka(int la, int k) { return _peaka[k * _g.nxya() + la]; };

    void expflux13(int l, int k, double* flux, double* phis, double* jnet);

	double getFxy();
	double getFq();
	double getFr();

    int& fxyb() { return _fxyb; };
    int& fxyt() { return _fxyt; };
	
};

