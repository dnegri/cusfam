#pragma once
#include <cuda_runtime.h>

__constant__ extern int		d_ng;
__constant__ extern int		d_ng2;
__constant__ extern float	d_rng;
__constant__ extern int		d_nxy;
__constant__ extern int		d_nz;
__constant__ extern int		d_nxyz;
__constant__ extern int		d_nsurf;
__constant__ extern int*	d_neibr;
__constant__ extern int*	d_neibz;
__constant__ extern int*	d_neib;
__constant__ extern float*	d_hmesh;

#define d_ptr3(var,ig,l,k)			(var[k*(d_nxy*d_ng)+l*d_ng+ig])
#define d_ptr4(var,ig,l,k,idir)		(var[NDIRMAX*(d_nz*d_nxy*d_ng)+k*(d_nxy*d_ng)+l*d_ng+ig])

