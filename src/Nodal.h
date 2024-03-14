#pragma once
#include "pch.h"
#include "Geometry.h"
#include "CrossSection.h"

#define m011   0.666666667
#define m022   0.4
#define m033   0.285714286
#define m044   0.222222222
#define m220   6.
#define rm220  0.166666667
#define m240   20.
#define m231   10.
#define m242   14.

class Nodal : public Managed {
protected:
	Geometry& _g;

	int* _ng;
	int* _ng2;
	int* _nxyz;
	int* _nsurf;
	int* _symopt;
	int* _symang;

	double* _albedo;

	int* _neib;
	int* _lktosfc;
	double* _hmesh;

	int* _lklr;
	int* _idirlr;
	int* _sgnlr;

	double* _xstf;
	double* _xsdf;
	double* _xsnf;
	double* _chif;
	double* _xssf;
	double* _xsadf;

	double* _trlcff0;
	double* _trlcff1;
	double* _trlcff2;
	double* _eta1;
	double* _eta2;
	double* _mu;
	double* _tau;


	double* _m260;
	double* _m251;
	double* _m253;
	double* _m262;
	double* _m264;

	double* _diagDI;
	double* _diagD;
	double* _matM;
	double* _matMI;
	double* _matMs;
	double* _matMf;

	double* _dsncff2;
	double* _dsncff4;
	double* _dsncff6;

	double* _jnet;
	double* _flux;
	double* _phis;
	double _reigv;
public:
	int nmaxswp;
	int nlupd;
	int nlswp;

public:
	Nodal(Geometry& g);
	virtual ~Nodal();

	virtual void init() = 0;
	virtual void reset(CrossSection& xs, const double& reigv, double* jnet, double* phif, double* phis) = 0;
	virtual void drive(double* jnet) = 0;


	void updateConstant(const int& lk);
	void updateMatrix(const int& lk);
	void trlcffbyintg(double* avgtrl3, double* hmesh3, double& trlcff1, double& trlcff2);
	void caltrlcff0(const int& lk);
	void caltrlcff12(const int& lk);
	void calculateEven(const int& lk);
	void calculateJnet(const int& ls);
	void calculateJnet1n(const int& ls, const int& lr, const float& alb);
	void calculateJnet2n(const int& ls);

	inline int& ng() { return *_ng; };
	inline int& ng2() { return *_ng2; };
	inline int& nxyz() { return *_nxyz; };
	inline int& nsurf() { return *_nsurf; };

	inline double& phis(const int& ig, const int& lks) { return _phis[(lks)* ng() + ig]; };

};