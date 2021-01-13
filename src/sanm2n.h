#pragma once
#include "pch.h"
#include "CrossSection.h"

__device__ const float m011 = 2. / 3.;
__device__ const float m022 = 2. / 5.;
__device__ const float m033 = 2. / 7.;
__device__ const float m044 = 2. / 9.;
__device__ const float m220 = 6.;
__device__ const float rm220 = 1 / 6.;
__device__ const float m240 = 20.;
__device__ const float m231 = 10.;
__device__ const float m242 = 14.;
__device__ const int d_ng = 2;
__device__ const int d_ng2 = 4;
__device__ const float d_rng = 0.5;


__host__ __device__ void
sanm2n_reset(const int &lk, const int& ng, const int& ng2, int &nxyz, float *hmesh, XS_PRECISION *xstf, XS_PRECISION *xsdf, float *eta1,
             float *eta2, float *m260, float *m251, float *m253, float *m262, float *m264, float *diagD,
             float *diagDI);

__host__ __device__ void
sanm2n_resetMatrix(const int &lk, const int& ng, const int& ng2, int &nxyz, double &reigv, XS_PRECISION *xstf, XS_PRECISION *xsnff,
                   XS_PRECISION *xschif, XS_PRECISION *xssf, float *matMs, float *matMf, float *matM);

__host__ __device__ void
sanm2n_prepareMatrix(const int &lk, const int& ng, const int& ng2, int &nxyz, float *m251, float *m253, float *diagD, float *diagDI,
                     float *matM, float *matMI, float *tau, float *mu);


__host__ __device__ void sanm2n_trlcffbyintg(float *avgtrl3, float *hmesh3, float &trlcff1, float &trlcff2);

__host__ __device__ void
sanm2n_calculateTransverseLeakage(const int &lk, const int& ng, const int& ng2, int &nxyz, int *lktosfc, int *idirlr, int *neib, float *hmesh,
                                  float *albedo, float *jnet, float *trlcff0, float *trlcff1, float *trlcff2);

__host__ __device__ void
sanm2n_calculateEven(const int &lk, const int& ng, const int& ng2, int &nxyz, float *m260, float *m262, float *m264, float *diagD,
                     float *diagDI, float *matM, double *, float *trlcff0, float *trlcff2, float *dsncff2,
                     float *dsncff4, float *dsncff6);

__host__ __device__ void sanm2n_calculateJnet(const int& ls, const int& ng, const int& ng2, int& nsurf, int* lklr, int* idirlr, int* sgnlr, float* albedo, float* hmesh, XS_PRECISION* xsadf, float* m251, float* m253, float* m260, float* m262, float* m264, float* diagD, float* diagDI, float* matM, float* matMI, double* flux, float* trlcff0, float* trlcff1, float* trlcff2, float* mu, float* tau, float* eta1, float* eta2, float* dsncff2, float* dsncff4, float* dsncff6, float* jnet);

__host__ __device__ void sanm2n_calculateJnet2n(const int &ls, const int& ng, const int& ng2, int &nsurf, int *lklr, int *idirlr, int *sgnlr, float *hmesh,
                                     XS_PRECISION *xsadf, float *m260, float *m262, float *m264, float *diagD,
                                     float *diagDI, float *matM, float *matMI, double *flux, float *trlcff0,
                                     float *trlcff1, float *trlcff2, float *mu, float *tau, float *eta1,
                                     float *eta2, float *dsncff2, float *dsncff4, float *dsncff6, float *jnet);

__host__ __device__ void sanm2n_calculateJnet1n(const int& ls, const int& lr, const int& ng, const int& ng2, int& nsurf, int* lklr, int* idirlr, int* sgnlr, float* hmesh, const float& albedo, XS_PRECISION* xsadf, float* m251, float* m253, float* diagD, float* matM, double* flux, float* trlcff1, float* eta1, float* eta2, float* dsncff2, float* dsncff4, float* dsncff6, float* jnet);
