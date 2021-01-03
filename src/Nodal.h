#pragma once

#include "pch.h"

class Nodal {
private:
	int _ng, _nxy, _nz, _ndir;
	float* d_trlcff0;
	float* d_trlcff1;
	float* d_trlcff2;
	float* d_jnet;
	float* d_eta1;
	float* d_eta2;
	float	m011 = 2. / 3.,
			m022 = 2. / 5.,
			m033 = 2. / 7.,
			m044 = 2. / 9.,
			m220 = 6.,
			rm220 = 1 / 6.,
			m240 = 20.,
			m231 = 10.,
			m242 = 14.;


	float* d_m260;
	float* d_m251;
	float* d_m253;
	float* d_m262;
	float* d_m264;

	float* d_xstf;
	float* d_xsdf;
	float* d_diagDI;
	float* d_diagD;
	float* d_matM;
	float* d_matMs;
	float* d_matMf;

	dim3 _blocks;
	dim3 _threads;


public:
	int nmaxswp;
	int nlupd;
	int nlswp;

public:
	Nodal();
	virtual ~Nodal();

	void reset();

	void drive();
	void calculateTransverseLeakage();
};