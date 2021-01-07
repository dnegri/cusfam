#pragma once

#include "pch.h"
#include <cuda_runtime.h>

#define d_neib(lr,idir,lk)	  (d_neib[lk*NEWSBT + idir*LR + lr])
#define d_lktosfc(lr,idir,lk) (d_lktosfc[lk*NEWSBT + idir*LR + lr])
#define d_jnet(ig,lks)	    (d_jnet[lks*d_ng + ig])
#define d_trlcff0(ig,lkd)	(d_trlcff0[lkd*d_ng + ig])
#define d_trlcff1(ig,lkd)	(d_trlcff1[lkd*d_ng + ig])
#define d_trlcff2(ig,lkd)	(d_trlcff2[lkd*d_ng + ig])
#define d_hmesh(idir,lk)		(d_hmesh[lk*NDIRMAX + idir])
#define d_lkg3(ig,l,k)		(k*(d_nxy*d_ng)+l*d_ng+ig)
#define d_lkg2(ig,lk)			(lk*d_ng+ig)

#define d_eta1(ig,lkd)	(d_eta1[lkd*d_ng + ig])
#define d_eta2(ig,lkd)	(d_eta2[lkd*d_ng + ig])
#define d_m260(ig,lkd)	(d_m260[lkd*d_ng + ig])
#define d_m251(ig,lkd)	(d_m251[lkd*d_ng + ig])
#define d_m253(ig,lkd)	(d_m253[lkd*d_ng + ig])
#define d_m262(ig,lkd)	(d_m262[lkd*d_ng + ig])
#define d_m264(ig,lkd)	(d_m264[lkd*d_ng + ig])
#define d_diagD(ig,lkd)	(d_diagD[lkd*d_ng + ig])
#define d_diagDI(ig,lkd)	(d_diagDI[lkd*d_ng + ig])
#define d_mu(i,j,lkd)	(d_mu[lkd*d_ng2 + j*d_ng + i])
#define d_tau(i,j,lkd)	(d_tau[lkd*d_ng2 + j*d_ng + i])
#define d_matM(i,j,lk)	(d_matM[lk*d_ng2 + j*d_ng + i])
#define d_matMI(i,j,lk)	(d_matMI[lk*d_ng2 + j*d_ng + i])
#define d_matMs(i,j,lk)	(d_matMs[lk*d_ng2 + j*d_ng + i])
#define d_matMf(i,j,lk)	(d_matMf[lk*d_ng2 + j*d_ng + i])
#define d_xssf(i,j,lk)	(d_xssf[lk*d_ng2 + j*d_ng + i])
#define d_xsadf(ig,lk)	(d_xsadf[lk*d_ng + ig])
#define d_flux(ig,lk)	(d_flux[lk*d_ng+ig])

#define d_dsncff2(ig,lkd) (d_dsncff2[lkd*d_ng + ig])
#define d_dsncff4(ig,lkd) (d_dsncff4[lkd*d_ng + ig])
#define d_dsncff6(ig,lkd) (d_dsncff6[lkd*d_ng + ig])


__constant__ extern float m011 = 2. / 3.;
__constant__ extern float m022 = 2. / 5.;
__constant__ extern float m033 = 2. / 7.;
__constant__ extern float m044 = 2. / 9.;
__constant__ extern float m220 = 6.;
__constant__ extern float rm220 = 1 / 6.;
__constant__ extern float m240 = 20.;
__constant__ extern float m231 = 10.;
__constant__ extern float m242 = 14.;

__constant__ extern int		d_ng = 2;
__constant__ extern int		d_ng2 = 4;
__constant__ extern float	d_rng = 0.5;



class Nodal {
public:
	int		d_nxy;
	int		d_nz;
	int		d_nxyz;
	int		d_nsurf;
	int*	d_neib;
	float*	d_hmesh;

	int* d_lklr;
	int* d_idirlr;
	int* d_sgnlr;
	int* d_lktosfc;

	float* d_trlcff0;
	float* d_trlcff1;
	float* d_trlcff2;
	double* d_flux;
	float* d_jnet;
	float* d_eta1;
	float* d_eta2;
	float* d_mu;
	float* d_tau;


	float* d_m260;
	float* d_m251;
	float* d_m253;
	float* d_m262;
	float* d_m264;

	float* d_xstf;
	float* d_xsdf;
	float* d_xsnff;
	float* d_xschif;
	float* d_xssf;
	float* d_xsadf;
	float* d_diagDI;
	float* d_diagD;
	float* d_matM;
	float* d_matMI;
	float* d_matMs;
	float* d_matMf;

	float* d_dsncff2;
	float* d_dsncff4;
	float* d_dsncff6;

	double* d_reigv;

	dim3 _blocks, _blocks_sfc;
	dim3 _threads, _threads_sfc;


public:
	int nmaxswp;
	int nlupd;
	int nlswp;

public:
	Nodal();
	void init();
	virtual ~Nodal();

	void reset();

	void drive();
};