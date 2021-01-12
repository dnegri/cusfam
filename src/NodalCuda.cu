#include "NodalCuda.h"
#include "helper_cuda.h"

__device__ const float m011 = 2. / 3.;
__device__ const float m022 = 2. / 5.;
__device__ const float m033 = 2. / 7.;
__device__ const float m044 = 2. / 9.;
__device__ const float m220 = 6.;
__device__ const float rm220 = 1 / 6.;
__device__ const float m240 = 20.;
__device__ const float m231 = 10.;
__device__ const float m242 = 14.;
__device__ const int		d_ng = 2;
__device__ const int		d_ng2 = 4;
__device__ const float	d_rng = 0.5;

static float* temp;

#define d_jnet(ig,lks)	    (d_jnet[lks*d_ng + ig])
#define d_trlcff0(ig,lkd)	(d_trlcff0[lkd*d_ng + ig])
#define d_trlcff1(ig,lkd)	(d_trlcff1[lkd*d_ng + ig])
#define d_trlcff2(ig,lkd)	(d_trlcff2[lkd*d_ng + ig])
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

#define d_hmesh(idir,lk)		(d_hmesh[lk*NDIRMAX+idir])
#define d_lktosfc(lr,idir,lk)	(d_lktosfc[(lk*NDIRMAX+idir)*LR + lr])
#define d_idirlr(lr,ls)			(d_idirlr[ls*LR + lr])
#define d_neib(lr, idir, lk)	(d_neib[(lk*NDIRMAX+idir)*LR + lr])
#define d_albedo(lr,idir)		(d_albedo[idir*LR + lr])

__global__ void reset(int& d_nxyz, float* d_hmesh, XS_PRECISION* d_xstf, XS_PRECISION* d_xsdf, float* d_eta1, float* d_eta2, float* d_m260, float* d_m251, float* d_m253, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= d_nxyz) return;

	int lkd0 = lk * NDIRMAX;
	int lkg0 = lk * d_ng;

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		int lkd = lkd0 + idir;

		for (size_t ig = 0; ig < d_ng; ig++)
		{
			int lkg = lkg0 + ig;
			auto kp2 = d_xstf[lkg] * d_hmesh(idir, lk) * d_hmesh(idir, lk) / (4 * d_xsdf[lkg]);
			auto kp = sqrt(kp2);
			auto kp3 = kp2 * kp;
			auto kp4 = kp2 * kp2;
			auto kp5 = kp2 * kp3;
			auto rkp = 1 / kp;
			auto rkp2 = rkp * rkp;
			auto rkp3 = rkp2 * rkp;
			auto rkp4 = rkp2 * rkp2;
			auto rkp5 = rkp2 * rkp3;
			auto sinhkp = sinh(kp);
			auto coshkp = cosh(kp);

			//calculate coefficient of basic functions P5and P6
			auto bfcff0 = -sinhkp * rkp;
			auto bfcff2 = -5 * (-3 * kp * coshkp + 3 * sinhkp + kp2 * sinhkp) * rkp3;
			auto bfcff4 = -9. * (-105 * kp * coshkp - 10 * kp3 * coshkp + 105 * sinhkp + 45 * kp2 * sinhkp + kp4 * sinhkp) * rkp5;
			auto bfcff1 = -3 * (kp * coshkp - sinhkp) * rkp2;
			auto bfcff3 = -7 * (15 * kp * coshkp + kp3 * coshkp - 15 * sinhkp - 6 * kp2 * sinhkp) * rkp4;

			auto oddtemp = 1 / (sinhkp + bfcff1 + bfcff3);
			auto eventemp = 1 / (coshkp + bfcff0 + bfcff2 + bfcff4);

			//eta1, eta2
			d_eta1(ig, lkd) = (kp * coshkp + bfcff1 + 6 * bfcff3) * oddtemp;
			d_eta2(ig, lkd) = (kp * sinhkp + 3 * bfcff2 + 10 * bfcff4) * eventemp;

			//set to variables that depends on node properties by integrating of Pi* pj over - 1 ~1
			d_m260(ig, lkd) = 2 * d_eta2(ig, lkd);
			d_m251(ig, lkd) = 2 * (kp * coshkp - sinhkp + 5 * bfcff3) * oddtemp;
			d_m253(ig, lkd) = 2 * (kp * (15 + kp2) * coshkp - 3 * (5 + 2 * kp2) * sinhkp) * oddtemp * rkp2;
			d_m262(ig, lkd) = 2 * (-3 * kp * coshkp + (3 + kp2) * sinhkp + 7 * kp * bfcff4) * eventemp * rkp;
			d_m264(ig, lkd) = 2 * (-5 * kp * (21 + 2 * kp2) * coshkp + (105 + 45 * kp2 + kp4) * sinhkp) * eventemp * rkp3;
			if (d_m264(ig, lkd) == 0.0) d_m264(ig, lkd) = 1.e-10;

			d_diagD(ig, lkd) = 4 * d_xsdf[lkg] / (d_hmesh(idir, lk) * d_hmesh(idir, lk));
			d_diagDI(ig, lkd) = 1.0 / d_diagD(ig, lkd);
		}
	}
}

__global__ void resetMatrix(int& d_nxyz, double& d_reigv, XS_PRECISION* d_xstf, XS_PRECISION* d_xsnff, XS_PRECISION* d_xschif, XS_PRECISION* d_xssf, float* d_matMs, float* d_matMf, float* d_matM) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;

	if (lk >= d_nxyz) return;
	int lkg0 = lk * d_ng;

	for (size_t ige = 0; ige < d_ng; ige++)
	{
		for (size_t igs = 0; igs < d_ng; igs++)
		{
			d_matMs(igs, ige, lk) = -d_xssf(igs, ige, lk);
			d_matMf(igs, ige, lk) = d_xschif[lkg0 + ige] * d_xsnff[lkg0 + igs];
		}
		d_matMs(ige, ige, lk) += d_xstf[lkg0 + ige];

		for (size_t igs = 0; igs < d_ng; igs++)
		{
			d_matM(igs, ige, lk) = d_matMs(igs, ige, lk) - d_reigv * d_matMf(igs, ige, lk);
		}
	}

}

__global__ void prepareMatrix(int& d_nxyz, float* d_m251, float* d_m253, float* d_diagD, float* d_diagDI, float* d_matM, float* d_matMI, float* d_tau, float* d_mu) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= d_nxyz) return;
	int lkd0 = lk * NDIRMAX;

	auto det = d_matM(0, 0, lk) * d_matM(1, 1, lk) - d_matM(1, 0, lk) * d_matM(0, 1, lk);

	if (abs(det) < 1.E-10) {
		auto rdet = 1 / det;
		d_matMI(0, 0, lk) = rdet * d_matM(1, 1, lk);
		d_matMI(1, 0, lk) = -rdet * d_matM(1, 0, lk);
		d_matMI(0, 1, lk) = -rdet * d_matM(0, 1, lk);
		d_matMI(1, 1, lk) = rdet * d_matM(0, 0, lk);
	}
	else {
		d_matMI(0, 0, lk) = 0;
		d_matMI(1, 0, lk) = 0;
		d_matMI(0, 1, lk) = 0;
		d_matMI(1, 1, lk) = 0;
	}

	auto rm011 = 1. / m011;

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		auto lkd = lkd0 + idir;

		float tempz[2][2] = { 0.0 };

		for (size_t igd = 0; igd < d_ng; igd++)
		{
			auto tau1 = m033 * (d_diagDI(igd, lkd) / d_m253(igd, lkd));

			tempz[igd][igd] = tempz[igd][igd] + m231;

			for (size_t igs = 0; igs < d_ng; igs++)
			{
				d_tau(igs, igd, lkd) = tau1 * d_matM(igs, igd, lk);

				// mu=m011_inv*M_inv*D*(m231*I+m251*tau)
				tempz[igs][igd] += d_m251(igd, lkd) * d_tau(igs, igd, lkd);

				// mu=m011_inv*M_inv*D*(m231*I+m251*tau)
				tempz[igs][igd] *= d_diagD(igd, lkd);
			}
		}

		// mu=m011_inv*M_inv*D*(m231*I+m251*tau)
		d_mu(0, 0, lkd) = rm011 * (d_matMI(0, 0, lk) * tempz[0][0] + d_matMI(1, 0, lk) * tempz[0][1]);
		d_mu(1, 0, lkd) = rm011 * (d_matMI(0, 0, lk) * tempz[1][0] + d_matMI(1, 0, lk) * tempz[1][1]);
		d_mu(0, 1, lkd) = rm011 * (d_matMI(0, 1, lk) * tempz[0][0] + d_matMI(1, 1, lk) * tempz[0][1]);
		d_mu(1, 1, lkd) = rm011 * (d_matMI(0, 1, lk) * tempz[1][0] + d_matMI(1, 1, lk) * tempz[1][1]);


	}

}



__device__ void trlcffbyintg(float* avgtrl3, float* hmesh3, float& trlcff1, float& trlcff2) {
	float sh[4];

	float rh = (1 / ((hmesh3[LEFT] + hmesh3[CENTER] + hmesh3[RIGHT]) * (hmesh3[LEFT] + hmesh3[CENTER]) * (hmesh3[CENTER] + hmesh3[RIGHT])));
	sh[0] = (2 * hmesh3[LEFT] + hmesh3[CENTER]) * (hmesh3[LEFT] + hmesh3[CENTER]);
	sh[1] = hmesh3[LEFT] + hmesh3[CENTER];
	sh[2] = (hmesh3[CENTER] + 2 * hmesh3[RIGHT]) * (hmesh3[CENTER] + hmesh3[RIGHT]);
	sh[3] = hmesh3[CENTER] + hmesh3[RIGHT];

	if (hmesh3[LEFT] == 0.0) {
		trlcff1 = 0.125 * (5. * avgtrl3[CENTER] + avgtrl3[RIGHT]);
		trlcff2 = 0.125 * (-3. * avgtrl3[CENTER] + avgtrl3[RIGHT]);
	}
	else if (hmesh3[RIGHT] == 0.0) {
		trlcff1 = -0.125 * (5. * avgtrl3[CENTER] + avgtrl3[LEFT]);
		trlcff2 = 0.125 * (-3. * avgtrl3[CENTER] + avgtrl3[LEFT]);
	}
	else {
		trlcff1 = 0.5 * rh * hmesh3[CENTER] * ((avgtrl3[CENTER] - avgtrl3[LEFT]) * sh[2] + (avgtrl3[RIGHT] - avgtrl3[CENTER]) * sh[0]);
		trlcff2 = 0.5 * rh * (hmesh3[CENTER] * hmesh3[CENTER]) * ((avgtrl3[LEFT] - avgtrl3[CENTER]) * sh[3] + (avgtrl3[RIGHT] - avgtrl3[CENTER]) * sh[1]);
	}
}

__global__ void calculateTransverseLeakage(int& d_nxyz, int* d_lktosfc, int* d_idirlr, int* d_neib, float* d_hmesh, float* d_albedo, float* d_jnet, float* d_trlcff0, float* d_trlcff1, float* d_trlcff2)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= d_nxyz) return;

	int lkd0 = lk * NDIRMAX;

	float avgjnet[NDIRMAX];

	for (size_t ig = 0; ig < d_ng; ig++)
	{
		for (size_t idir = 0; idir < NDIRMAX; idir++)
		{
			auto lsl = d_lktosfc(LEFT, idir, lk);
			auto lsr = d_lktosfc(RIGHT, idir, lk);

			avgjnet[idir] = (d_jnet(ig, lsr) - d_jnet(ig, lsl)) * d_hmesh(idir, lk);

			d_trlcff0(ig, lkd0 + XDIR) = avgjnet[YDIR] + avgjnet[ZDIR];
			d_trlcff0(ig, lkd0 + YDIR) = avgjnet[XDIR] + avgjnet[ZDIR];
			d_trlcff0(ig, lkd0 + ZDIR) = avgjnet[XDIR] + avgjnet[YDIR];
		}
	}

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		int lkd = lkd0 + idir;

		for (size_t ig = 0; ig < d_ng; ig++)
		{
			float avgtrl3[LRC] = { 0.0 };
			float hmesh3[LRC] = { 0.0 };
			hmesh3[CENTER] = d_hmesh(idir, lk);
			avgtrl3[CENTER] = d_trlcff0(ig, lkd);

			int lkl = d_neib(LEFT, idir, lk);
			int lsl = d_lktosfc(LEFT, idir, lk);
			int idirl = d_idirlr(LEFT, lsl);

			if (lkl < 0 && d_albedo(LEFT, idir) == 0) {
				hmesh3[LEFT] = hmesh3[CENTER];
				avgtrl3[LEFT] = avgtrl3[CENTER];
			}
			else {
				hmesh3[LEFT] = d_hmesh(idirl,lkl);
				avgtrl3[LEFT] = d_trlcff0(ig, lkd0+idirl);
			}

			int lkr = d_neib(RIGHT, idir, lk);
			int lsr = d_lktosfc(RIGHT, idir, lk);
			int idirr = d_idirlr(RIGHT, lsr);

			if (lkr < 0 && d_albedo(RIGHT, idir) == 0) {
				hmesh3[RIGHT] = hmesh3[CENTER];
				avgtrl3[RIGHT] = avgtrl3[CENTER];
			}
			else {
				hmesh3[RIGHT] = d_hmesh(idirr, lkr);
				avgtrl3[RIGHT] = d_trlcff0(ig, lkd0 + idirr);
			}


			trlcffbyintg(avgtrl3, hmesh3, d_trlcff1(ig, lkd), d_trlcff2(ig, lkd));
			//printf("trlcff12: %e %e \n", d_trlcff1(ig, lkd), d_trlcff2(ig, lkd));
		}

	}
}

__global__ void calculateEven(int& d_nxyz, float* d_m260, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI, float* d_matM, double* d_flux, float* d_trlcff0, float* d_trlcff2, float* d_dsncff2, float* d_dsncff4, float* d_dsncff6)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	if (lk >= d_nxyz) return;

	int lkd0 = lk * NDIRMAX;

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		auto lkd = lkd0 + idir;
		float at2[2][2], a[2][2], rm4464[2], bt1[2], bt2[2], b[2];

		for (size_t igd = 0; igd < d_ng; igd++)
		{
			rm4464[igd] = m044 / d_m264(igd, lkd);
			auto mu2 = rm4464[igd] * d_m260(igd, lkd) * d_diagDI(igd, lkd);

			for (size_t igs = 0; igs < d_ng; igs++)
			{
				at2[igs][igd] = m022 * rm220 * mu2 * d_matM(igs, igd, lk);
			}
			at2[igd][igd] += m022 * rm220 * m240;
		}

		printf("at2 : %e %e %e %e\n", at2[0][0], at2[1][0], at2[0][1], at2[1][1]);

		for (size_t igd = 0; igd < d_ng; igd++)
		{
			auto mu1 = rm4464[igd] * d_m262(igd, lkd);
			for (size_t igs = 0; igs < d_ng; igs++)
			{
				a[igs][igd] = mu1 * d_matM(igs, igd, lk) + d_matM(0, igd, lk) * at2[igs][0] + d_matM(1, igd, lk) * at2[igs][1];
			}
			a[igd][igd] += d_diagD(igd, lkd) * m242;
			bt2[igd] = 2 * (d_matM(0, igd, lk) * d_flux(0, lk) + d_matM(1, igd, lk) * d_flux(0, lk) + d_trlcff0(igd, lkd));
			bt1[igd] = m022 * rm220 * d_diagDI(igd, lkd) * bt2[igd];
		}
		printf("bt1 bt2 : %e %e %e %e\n", bt1[0], bt1[1], bt2[0], bt2[1]);

		for (size_t ig = 0; ig < d_ng; ig++)
		{
			b[ig] = m220 * d_trlcff2(ig, lkd) + d_matM(0, ig, lk) * bt1[0] + d_matM(1, ig, lk) * bt1[1];
		}

		auto rdet = 1 / (a[0][0] * a[1][1] - a[1][0] * a[0][1]);
		d_dsncff4(0, lkd) = rdet * (a[1][1] * b[0] - a[1][0] * b[1]);
		d_dsncff4(1, lkd) = rdet * (a[0][0] * b[1] - a[0][1] * b[0]);

		for (size_t ig = 0; ig < d_ng; ig++)
		{
			d_dsncff6(ig, lkd) = d_diagDI(ig, lkd) * rm4464[ig] * (d_matM(0, ig, lk) * d_dsncff4(0, lkd) + d_matM(1, ig, lk) * d_dsncff4(1, lkd));
			d_dsncff2(ig, lkd) = rm220 * (d_diagDI(ig, lkd) * bt2[ig] - m240 * d_dsncff4(ig, lkd) - d_m260(ig, lkd) * d_dsncff6(ig, lkd));
		}
	}
}

__global__ void calculateJnet(int& d_nsurf, int* lklr, int* idirlr, int* sgnlr, float* d_hmesh, XS_PRECISION* d_xsadf, float* d_m260, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI, float* d_matM, float* d_matMI, double* d_flux, float* d_trlcff0, float* d_trlcff1, float* d_trlcff2, float* d_mu, float* d_tau, float* d_eta1, float* d_eta2, float* d_dsncff2, float* d_dsncff4, float* d_dsncff6, float* d_jnet)
{
	int ls = threadIdx.x + blockIdx.x * blockDim.x;
	if (ls >= d_nsurf) return;

	int lsfclr = ls * LR;

	int lkl = lklr[lsfclr + LEFT];
	int lkr = lklr[lsfclr + RIGHT];
	int idirl = idirlr[lsfclr + LEFT];
	int idirr = idirlr[lsfclr + RIGHT];
	int sgnl = sgnlr[lsfclr + LEFT];
	int sgnr = sgnlr[lsfclr + RIGHT];
	int lkdl = lkl * NDIRMAX + idirl;
	int lkdr = lkr * NDIRMAX + idirr;

	float adf[2][LR], d_diagDj[2][LR], tempz[2][2], tempzI[2][2], zeta1[2][2], zeta2[2], bfc[2], mat1g[2][2];

	for (size_t ig = 0; ig < d_ng; ig++)
	{
		adf[ig][LEFT] = d_xsadf(ig, lkl);
		adf[ig][RIGHT] = d_xsadf(ig, lkr);
		d_diagDj[ig][LEFT] = 0.5 * d_hmesh(idirl, lkl) * d_diagD(ig, lkdl);
		d_diagDj[ig][RIGHT] = 0.5 * d_hmesh(idirr, lkr) * d_diagD(ig, lkdr);
	}

	//zeta1 = (d_mur + I + d_taur)_inv * (d_mul + I + d_taul)
	tempz[0][0] = (d_mu(0, 0, lkdr) + d_tau(0, 0, lkdr) + 1) * adf[0][RIGHT];
	tempz[1][0] = (d_mu(1, 0, lkdr) + d_tau(1, 0, lkdr)) * adf[0][RIGHT];
	tempz[0][1] = (d_mu(0, 1, lkdr) + d_tau(0, 1, lkdr)) * adf[1][RIGHT];
	tempz[1][1] = (d_mu(1, 1, lkdr) + d_tau(1, 1, lkdr) + 1) * adf[1][RIGHT];

	auto rdet = 1 / (tempz[0][0] * tempz[1][1] - tempz[1][0] * tempz[0][1]);
	tempzI[0][0] = rdet * tempz[1][1];
	tempzI[1][0] = -rdet * tempz[1][0];
	tempzI[0][1] = -rdet * tempz[0][1];
	tempzI[1][1] = rdet * tempz[0][0];

	tempz[0][0] = (d_mu(0, 0, lkdl) + d_tau(0, 0, lkdl) + 1) * adf[0][LEFT];
	tempz[1][0] = (d_mu(1, 0, lkdl) + d_tau(1, 0, lkdl)) * adf[0][LEFT];
	tempz[0][1] = (d_mu(0, 1, lkdl) + d_tau(0, 1, lkdl)) * adf[1][LEFT];
	tempz[1][1] = (d_mu(1, 1, lkdl) + d_tau(1, 1, lkdl) + 1) * adf[1][LEFT];

	zeta1[0][0] = tempzI[0][0] * tempz[0][0] + tempzI[1][0] * tempz[0][1];
	zeta1[1][0] = tempzI[0][0] * tempz[1][0] + tempzI[1][0] * tempz[1][1];
	zeta1[0][1] = tempzI[0][1] * tempz[0][0] + tempzI[1][1] * tempz[0][1];
	zeta1[1][1] = tempzI[0][1] * tempz[1][0] + tempzI[1][1] * tempz[1][1];

	for (size_t ig = 0; ig < d_ng; ig++)
	{
		bfc[ig] = adf[ig][RIGHT] * (d_dsncff2(ig, lkdr) + d_dsncff4(ig, lkdr) + d_dsncff6(ig, lkdr)
			+ d_flux(ig, lkr) + d_matMI(0, ig, lkr) * sgnr * d_trlcff1(0, lkdr)
			+ d_matMI(1, ig, lkr) * sgnr * d_trlcff1(1, lkdr))
			+ adf[ig][LEFT] * (-d_dsncff2(ig, lkdl) - d_dsncff4(ig, lkdl) - d_dsncff6(ig, lkdl)
				- d_flux(ig, lkl) + d_matMI(0, ig, lkl) * sgnl * d_trlcff1(0, lkdl)
				+ d_matMI(1, ig, lkl) * sgnl * d_trlcff1(1, lkdl));
	}

	for (size_t ig = 0; ig < d_ng; ig++)
	{
		zeta2[ig] = tempzI[0][ig] * bfc[0] + tempzI[1][ig] * bfc[1];
	}

	//tempz = d_mur + 6 * I + eta1 * d_taur
	tempz[0][0] = d_diagDj[0][RIGHT] * (d_mu(0, 0, lkdr) + 6 + d_eta1(0, lkdr) * d_tau(0, 0, lkdr));
	tempz[1][0] = d_diagDj[0][RIGHT] * (d_mu(1, 0, lkdr) + d_eta1(0, lkdr) * d_tau(1, 0, lkdr));
	tempz[0][1] = d_diagDj[1][RIGHT] * (d_mu(0, 1, lkdr) + d_eta1(1, lkdr) * d_tau(0, 1, lkdr));
	tempz[1][1] = d_diagDj[1][RIGHT] * (d_mu(1, 1, lkdr) + 6 + d_eta1(1, lkdr) * d_tau(1, 1, lkdr));


	//mat1g = d_mul + 6 * I + eta1 * d_taul - tempzI
	mat1g[0][0] = -d_diagDj[0][LEFT] * (d_mu(0, 0, lkdl) + 6 + d_eta1(0, lkdl) * d_tau(0, 0, lkdl)) - tempz[0][0] * zeta1[0][0] - tempz[1][0] * zeta1[0][1];
	mat1g[1][0] = -d_diagDj[0][LEFT] * (d_mu(1, 0, lkdl) + d_eta1(0, lkdl) * d_tau(1, 0, lkdl)) - tempz[0][0] * zeta1[1][0] - tempz[1][0] * zeta1[1][1];
	mat1g[0][1] = -d_diagDj[1][LEFT] * (d_mu(0, 1, lkdl) + d_eta1(1, lkdl) * d_tau(0, 1, lkdl)) - tempz[0][1] * zeta1[0][0] - tempz[1][1] * zeta1[0][1];
	mat1g[1][1] = -d_diagDj[1][LEFT] * (d_mu(1, 1, lkdl) + 6 + d_eta1(1, lkdl) * d_tau(1, 1, lkdl)) - tempz[0][1] * zeta1[1][0] - tempz[1][1] * zeta1[1][1];


	float bcc[2], vec1g[2];

	for (size_t ig = 0; ig < d_ng; ig++)
	{
		bcc[ig] = d_diagDj[ig][LEFT] * (3 * d_dsncff2(ig, lkdl) + 10 * d_dsncff4(ig, lkdl) + d_eta2(ig, lkdl) * d_dsncff6(ig, lkdl))
			+ d_diagDj[ig][RIGHT] * (3 * d_dsncff2(ig, lkdr) + 10 * d_dsncff4(ig, lkdr) + d_eta2(ig, lkdr) * d_dsncff6(ig, lkdr));
		vec1g[ig] = bcc[ig]
			- d_diagDj[ig][LEFT] * (d_matMI(0, ig, lkl) * sgnl * d_trlcff1(0, lkdl) + d_matMI(0, ig, lkl) * sgnl * d_trlcff1(0, lkdl))
			+ d_diagDj[ig][RIGHT] * (d_matMI(1, ig, lkr) * sgnr * d_trlcff1(1, lkdr) + d_matMI(1, ig, lkr) * sgnr * d_trlcff1(1, lkdr))
			- (tempz[0][ig] * zeta2[0] + tempz[1][ig] * zeta2[1]);

	}

	rdet = 1 / (mat1g[0][0] * mat1g[1][1] - mat1g[1][0] * mat1g[0][1]);
	auto tmp = mat1g[0][0];
	mat1g[0][0] = rdet * mat1g[1][1];
	mat1g[1][0] = -rdet * mat1g[1][0];
	mat1g[0][1] = -rdet * mat1g[0][1];
	mat1g[1][1] = rdet * tmp;

	float oddcff[3][2];

	oddcff[1][0] = zeta2[0] - (zeta1[0][0] * (mat1g[0][0] * vec1g[0] + mat1g[1][0] * vec1g[1])
		+ zeta1[1][0] * (mat1g[0][1] * vec1g[0] + mat1g[1][1] * vec1g[1]));
	oddcff[1][1] = zeta2[1] - (zeta1[0][1] * (mat1g[0][0] * vec1g[0] + mat1g[1][0] * vec1g[1])
		+ zeta1[1][1] * (mat1g[0][1] * vec1g[0] + mat1g[1][1] * vec1g[1]));

	oddcff[2][0] = d_tau(0, 0, lkdr) * oddcff[1][0] + d_tau(1, 0, lkdr) * oddcff[1][1];
	oddcff[2][1] = d_tau(0, 1, lkdr) * oddcff[1][0] + d_tau(1, 1, lkdr) * oddcff[1][1];

	oddcff[0][0] = d_mu(0, 0, lkdr) * oddcff[1][0] - d_matMI(0, 0, lkr) * sgnr * d_trlcff1(0, lkdr)
		+ d_mu(1, 0, lkdr) * oddcff[1][1] - d_matMI(1, 0, lkr) * sgnr * d_trlcff1(1, lkdr);
	oddcff[0][1] = d_mu(0, 1, lkdr) * oddcff[1][0] - d_matMI(0, 1, lkr) * sgnr * d_trlcff1(0, lkdr)
		+ d_mu(1, 1, lkdr) * oddcff[1][1] - d_matMI(1, 1, lkr) * sgnr * d_trlcff1(1, lkdr);

	for (size_t ig = 0; ig < d_ng; ig++)
	{
		d_jnet(ig, ls) = sgnr * d_hmesh(idirr, lkr) * 0.5 * d_diagD(ig, lkdr) * (
			-1.0 * oddcff[0][ig] + 3 * d_dsncff2(ig, lkdr) - 6 * oddcff[1][ig] + 10 * d_dsncff4(ig, lkdr)
			- d_eta1(ig, lkdr) * oddcff[2][ig] + d_eta2(ig, lkdr) * d_dsncff6(ig, lkdr));
	}

}

NodalCuda::NodalCuda(Geometry& g): Nodal(g)
{
	_ng = _g.ng();
	_ng2 = _ng * _ng;
	_nxyz = _g.nxyz();
	_nsurf = _g.nsurf();

	_d_blocks = dim3(_nxyz / NTHREADSPERBLOCK + 1, 1, 1);
	_d_threads = dim3(NTHREADSPERBLOCK, 1, 1);

	_d_blocks_sfc = dim3(_nsurf / NTHREADSPERBLOCK + 1, 1, 1);
	_d_threads_sfc = dim3(NTHREADSPERBLOCK, 1, 1);

	_jnet = new float[_nsurf * _ng];
	_flux = new double[_nxyz * _ng];


	checkCudaErrors(cudaMalloc((void**)&_d_symopt, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_d_symang, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_d_albedo, sizeof(float)*NDIRMAX*LR));
	checkCudaErrors(cudaMemcpy(_d_symopt, &_g.symopt(), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_symopt, &_g.symang(), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_albedo, &_g.albedo(0,0), sizeof(float) * NDIRMAX * LR, cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&_d_nxyz, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&_d_nsurf, sizeof(int)));
	checkCudaErrors(cudaMemcpy(_d_nxyz, &_nxyz, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_nsurf, &_nsurf, sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&_d_neib, sizeof(int) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_d_lktosfc, sizeof(int) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_d_hmesh, sizeof(float) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_d_idirlr, sizeof(int) * LR* _nsurf));
	checkCudaErrors(cudaMalloc((void**)&_d_sgnlr, sizeof(int) * LR * _nsurf));
	checkCudaErrors(cudaMalloc((void**)&_d_lklr, sizeof(int) * LR * _nsurf));

	checkCudaErrors(cudaMemcpy(_d_neib	, &_g.neib(0, 0)	, sizeof(int) * NEWSBT * _nxyz	, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_lktosfc	, &_g.lktosfc(0,0,0), sizeof(int) * NEWSBT * _nxyz	, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_hmesh	, &_g.hmesh(0, 0)	, sizeof(float) * NEWSBT * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_idirlr	, &_g.idirlr(0, 0)	, sizeof(int) * LR * _nsurf		, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_sgnlr	, &_g.sgnlr(0, 0)	, sizeof(int) * LR * _nsurf		, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_lklr	, &_g.lklr(0, 0)	, sizeof(int) * LR * _nsurf		, cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&_d_reigv, sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&_d_jnet, sizeof(float) * _nsurf * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_flux, sizeof(double) * _nxyz * _ng));

	checkCudaErrors(cudaMalloc((void**)&_d_trlcff0, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_trlcff1, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_trlcff2, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_eta1, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_eta2, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_mu, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_tau, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_m260, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_m251, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_m253, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_m262, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_m264, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_diagDI, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_diagD, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_dsncff2, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_dsncff4, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_dsncff6, sizeof(float) * _nxyz * NDIRMAX * _ng));

	checkCudaErrors(cudaMalloc((void**)&_d_xstf, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_xsdf, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_xsnff, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_xschif, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_d_xsadf, sizeof(XS_PRECISION) * _nxyz * _ng));

	checkCudaErrors(cudaMalloc((void**)&_d_xssf, sizeof(XS_PRECISION) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_d_matM, sizeof(float) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_d_matMI, sizeof(float) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_d_matMs, sizeof(float) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_d_matMf, sizeof(float) * _nxyz * _ng2));


}

NodalCuda::~NodalCuda()
{
}

void NodalCuda::init()
{
	temp = new float[10000];
}

void NodalCuda::reset(CrossSection& xs, double* reigv, double* jnet, double* phif)
{

	for (size_t ls = 0; ls < _nsurf; ls++)
	{
		int idirl = _g.idirlr(LEFT, ls);
		int idirr = _g.idirlr(RIGHT, ls);
		int lkl   = _g.lklr(LEFT, ls);
		int lkr   = _g.lklr(RIGHT, ls);
		int kl = lkl / _g.nxy();
		int ll = lkl % _g.nxy();
		int kr = lkr / _g.nxy();
		int lr = lkr % _g.nxy();


		for (size_t ig = 0; ig < _ng; ig++)
		{
			if (lkr < 0) {
				int idx =
					idirl * (_g.nz() * _g.nxy() * _g.ng() * LR)
					+ kl * (_g.nxy() * _g.ng() * LR)
					+ ll * (_g.ng() * LR)
					+ ig * LR
					+ RIGHT;
					this->jnet(ig, ls) = jnet[idx];
			}
			else {
				int idx =
					idirr * (_g.nz() * _g.nxy() * _g.ng() * LR)
					+ kr * (_g.nxy() * _g.ng() * LR)
					+ lr * (_g.ng() * LR)
					+ ig * LR
					+ LEFT;
				this->jnet(ig, ls) = jnet[idx];
			}
		}
	}

	int lk = -1;
	for (size_t k = 0; k < _g.nz(); k++)
	{
		for (size_t l = 0; l < _g.nxy(); l++)
		{
			lk++;
			for (size_t ig = 0; ig < _g.ng(); ig++)
			{
				int idx = (k + 1) *(_g.nxy()+1) * _g.ng() + (l + 1) * _g.ng() + ig;
				this->flux(ig, lk) = phif[idx];
			}
		}
	}

	_reigv = *reigv;
	checkCudaErrors(cudaMemcpy(_d_reigv, &_reigv, sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_flux, _flux, sizeof(double) * _nxyz * _ng, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(_d_xsnff, &xs.xsnf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_xsdf, &xs.xsdf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_xstf, &xs.xstf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_xschif, &xs.chif(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_xsadf,&xs.xsadf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_xssf, &xs.xssf(0, 0, 0), sizeof(XS_PRECISION) * _nxyz * _ng2, cudaMemcpyHostToDevice));

}

void NodalCuda::drive()
{
	::reset << <_d_blocks, _d_threads >> > (*_d_nxyz, _d_hmesh, _d_xstf, _d_xsdf, _d_eta1, _d_eta2, _d_m260, _d_m251, _d_m253, _d_m262, _d_m264, _d_diagD, _d_diagDI);
	::calculateTransverseLeakage << <_d_blocks, _d_threads >> > (*_d_nxyz, _d_lktosfc, _d_idirlr, _d_neib, _d_hmesh, _d_albedo, _d_jnet, _d_trlcff0, _d_trlcff1, _d_trlcff2);
	checkCudaErrors(cudaMemcpy(temp, _d_trlcff0, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _d_trlcff1, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _d_trlcff2, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));

	::resetMatrix << <_d_blocks, _d_threads >> > (*_d_nxyz, *_d_reigv, _d_xstf, _d_xsnff, _d_xschif, _d_xssf, _d_matMs, _d_matMf, _d_matM);

	checkCudaErrors(cudaMemcpy(temp, _d_matM, sizeof(float) * _nxyz * _ng2, cudaMemcpyDeviceToHost));

	::prepareMatrix << <_d_blocks, _d_threads >> > (*_d_nxyz, _d_m251, _d_m253, _d_diagD, _d_diagDI, _d_matM, _d_matMI, _d_tau, _d_mu);

	checkCudaErrors(cudaMemcpy(temp, _d_mu, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));


	::calculateEven << <_d_blocks, _d_threads >> > (*_d_nxyz, _d_m260, _d_m262, _d_m264, _d_diagD, _d_diagDI, _d_matM, _d_flux, _d_trlcff0, _d_trlcff2, _d_dsncff2, _d_dsncff4, _d_dsncff6);

	checkCudaErrors(cudaMemcpy(temp, _d_dsncff4, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _d_dsncff6, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, _d_dsncff2, sizeof(float) * _nxyz * NDIRMAX * _ng, cudaMemcpyDeviceToHost));

	::calculateJnet << <_d_blocks_sfc, _d_threads_sfc >> > (*_d_nsurf, _d_lklr, _d_idirlr, _d_sgnlr, _d_hmesh, _d_xsadf, _d_m260, _d_m262, _d_m264,
		_d_diagD, _d_diagDI, _d_matM, _d_matMI, _d_flux, _d_trlcff0, _d_trlcff1,
		_d_trlcff2, _d_mu, _d_tau, _d_eta1, _d_eta2, _d_dsncff2, _d_dsncff4, _d_dsncff6, _d_jnet);

	checkCudaErrors(cudaMemcpy(_jnet, _d_jnet,sizeof(float) * _nsurf * _ng, cudaMemcpyDeviceToHost));
}
