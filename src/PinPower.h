#pragma once
#include "pch.h"
#include "Geometry.h"

class PinPower
{
private:

	Geometry& _g;
	

	bool usemss = true;
	bool term15 = true;
	int  nmaxppr = 20;

	double     sqrt2 = 1.41421356,
		rsqrt2 = 0.70710678;

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

public:
	PinPower(Geometry& g);
	virtual ~PinPower();


	inline double& phicorn(const int& ig, const int& lc, const int& k) {
		return _phicorn[k*_g.ncorn()*_g.ng()+lc*_g.ng()+ig];
	}

	void calphicorn(SOL_VAR* flux, SOL_VAR* phis);
	void init();
	void calhomo();
	void caltrlz();

	
};

