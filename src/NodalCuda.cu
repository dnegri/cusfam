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

#define d_hmesh(idir,lk)	(d_hmesh[lk*NDIRMAX+idir])
#define d_lktosfc(lr,idir,lk)	(d_lktosfc[(lk*NDIRMAX+idir)*LR + lr])
#define d_neib(lr, idir, lk)	(d_neib[(lk*NDIRMAX+idir)*LR + lr])

__global__ void reset(float* d_hmesh, XS_PRECISION* d_xstf, XS_PRECISION* d_xsdf, float* d_eta1, float* d_eta2, float* d_m260, float* d_m251, float* d_m253, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
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

__global__ void resetMatrix(double& d_reigv, XS_PRECISION* d_xstf, XS_PRECISION* d_xsnff, XS_PRECISION* d_xschif, XS_PRECISION* d_xssf, float* d_matMs, float* d_matMf, float* d_matM) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	int lkg0 = lk * d_ng;

	for (size_t igd = 0; igd < d_ng; igd++)
	{
		for (size_t igs = 0; igs < d_ng; igs++)
		{
			d_matMs(igs, igd, lk) = -d_xssf(igs, igd, lk);
			d_matMf(igs, igd, lk) = d_xschif[lkg0 + igd] * d_xsnff[lkg0 + igs];
		}
		d_matMs(igd, igd, lk) += d_xstf[lkg0 + igd];
	}
}

__global__ void prepareMatrix(float* d_m251, float* d_m253, float* d_diagD, float* d_diagDI, float* d_matM, float* d_matMI, float* d_tau, float* d_mu) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
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

	auto rh = (1 / ((hmesh3[1] + hmesh3[0] + hmesh3[2]) * (hmesh3[1] + hmesh3[0]) * (hmesh3[0] + hmesh3[2])));
	sh[1] = (2 * hmesh3[1] + hmesh3[0]) * (hmesh3[1] + hmesh3[0]);
	sh[2] = hmesh3[1] + hmesh3[0];
	sh[3] = (hmesh3[0] + 2 * hmesh3[2]) * (hmesh3[0] + hmesh3[2]);
	sh[4] = hmesh3[0] + hmesh3[2];

	if (hmesh3[LEFT] == 0.0) {
		trlcff1 = 0.125 * (5. * avgtrl3[CENTER] + avgtrl3[RIGHT]);
		trlcff2 = 0.125 * (-3. * avgtrl3[CENTER] + avgtrl3[RIGHT]);
	}
	else if (hmesh3[RIGHT] == 0.0) {
		trlcff1 = -0.125 * (5. * avgtrl3[CENTER] + avgtrl3[LEFT]);
		trlcff2 = 0.125 * (-3. * avgtrl3[CENTER] + avgtrl3[LEFT]);
	}
	else {
		trlcff1 = 0.5 * rh * hmesh3[0] * ((avgtrl3[CENTER] - avgtrl3[LEFT]) * sh[3] + (avgtrl3[RIGHT] - avgtrl3[CENTER]) * sh[1]);
		trlcff2 = 0.5 * rh * (hmesh3[0] * hmesh3[0]) * ((avgtrl3[LEFT] - avgtrl3[CENTER]) * sh[4] + (avgtrl3[RIGHT] - avgtrl3[CENTER]) * sh[2]);
	}

}

__global__ void calculateTransverseLeakage(int* d_lktosfc, int* d_neib, float* d_hmesh, float* d_jnet, float* d_trlcff0, float* d_trlcff1, float* d_trlcff2)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	int lkd0 = lk * NDIRMAX;

	float avgjnet[NDIRMAX];

	for (size_t ig = 0; ig < d_ng; ig++)
	{

		for (size_t idir = 0; idir < NDIRMAX; idir++)
		{
			auto lksl = d_lktosfc(LEFT, idir, lk);
			auto lksr = d_lktosfc(RIGHT, idir, lk);

			avgjnet[idir] = (d_jnet(ig, lksr) - d_jnet(ig, lksl)) * d_hmesh(idir, lk);

			d_trlcff0(ig, lkd0 + XDIR) = avgjnet[YDIR] + avgjnet[ZDIR];
			d_trlcff0(ig, lkd0 + YDIR) = avgjnet[XDIR] + avgjnet[ZDIR];
			d_trlcff0(ig, lkd0 + ZDIR) = avgjnet[XDIR] + avgjnet[YDIR];
		}
	}

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		int lkd = lkd0 + idir;

		float avgtrl3[LRC] = { 0.0 }, hmesh3[LRC] = { 0.0 };

		hmesh3[CENTER] = d_hmesh(idir, lk);

		for (size_t lr = 0; lr < LR; lr++)
		{
			auto lnk = d_neib(lr, idir, lk);
			int lnkd = lnk * NDIRMAX + idir;
			hmesh3[lr] = d_hmesh(idir, lnk);

			for (size_t ig = 0; ig < d_ng; ig++)
			{
				avgtrl3[CENTER] = d_trlcff0(ig, lkd);
				avgtrl3[lr] = d_trlcff0(ig, lnkd);
				trlcffbyintg(avgtrl3, hmesh3, d_trlcff1(ig, lkd), d_trlcff2(ig, lkd));
			}
		}
	}
}

__global__ void calculateEven(float* d_m260, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI, float* d_matM, double* d_flux, float* d_trlcff0, float* d_trlcff2, float* d_dsncff2, float* d_dsncff4, float* d_dsncff6)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
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

__global__ void calculateJnet(int* lklr, int* idirlr, int* sgnlr, float* d_hmesh, XS_PRECISION* d_xsadf, float* d_m260, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI, float* d_matM, float* d_matMI, double* d_flux, float* d_trlcff0, float* d_trlcff1, float* d_trlcff2, float* d_mu, float* d_tau, float* d_eta1, float* d_eta2, float* d_dsncff2, float* d_dsncff4, float* d_dsncff6, float* d_jnet)
{
	int lsfc = threadIdx.x + blockIdx.x * blockDim.x;
	int lsfclr = lsfc * LR;

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
		d_jnet(ig, lsfc) = sgnr * d_hmesh(idirr, lkr) * 0.5 * d_diagD(ig, lkdr) * (
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

	_blocks = dim3(_nxyz / NTHREADSPERBLOCK + 1, 1, 1);
	_threads = dim3(NTHREADSPERBLOCK, 1, 1);

	_blocks_sfc = dim3(_nsurf / NTHREADSPERBLOCK + 1, 1, 1);
	_threads_sfc = dim3(NTHREADSPERBLOCK, 1, 1);

	checkCudaErrors(cudaMalloc((void**)&_neib, sizeof(int) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_lktosfc, sizeof(int) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_hmesh, sizeof(float) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_idirlr, sizeof(int) * LR* _nsurf));
	checkCudaErrors(cudaMalloc((void**)&_sgnlr, sizeof(int) * LR * _nsurf));
	checkCudaErrors(cudaMalloc((void**)&_lklr, sizeof(int) * LR * _nsurf));

	checkCudaErrors(cudaMemcpy(_neib	, &_g.neib(0, 0)	, sizeof(int) * NEWSBT * _nxyz	, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_lktosfc	, &_g.lktosfc(0,0,0), sizeof(int) * NEWSBT * _nxyz	, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_hmesh	, &_g.hmesh(0, 0)	, sizeof(float) * NEWSBT * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_idirlr	, &_g.idirlr(0, 0)	, sizeof(int) * LR * _nsurf		, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_sgnlr	, &_g.sgnlr(0, 0)	, sizeof(int) * LR * _nsurf		, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_lklr	, &_g.lklr(0, 0)	, sizeof(int) * LR * _nsurf		, cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&_trlcff0, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_trlcff1, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_trlcff2, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_jnet, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_eta1, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_eta2, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_mu, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_tau, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_m260, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_m251, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_m253, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_m262, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_m264, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_diagDI, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_diagD, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dsncff2, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dsncff4, sizeof(float) * _nxyz * NDIRMAX * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dsncff6, sizeof(float) * _nxyz * NDIRMAX * _ng));

	checkCudaErrors(cudaMalloc((void**)&_xstf, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_xsdf, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_xsnff, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_xschif, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_xsadf, sizeof(XS_PRECISION) * _nxyz * _ng));

	checkCudaErrors(cudaMalloc((void**)&_xssf, sizeof(XS_PRECISION) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_matM, sizeof(float) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_matMI, sizeof(float) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_matMs, sizeof(float) * _nxyz * _ng2));
	checkCudaErrors(cudaMalloc((void**)&_matMf, sizeof(float) * _nxyz * _ng2));


	checkCudaErrors(cudaMalloc((void**)&_neib, sizeof(int) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_lktosfc, sizeof(int) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_hmesh, sizeof(float) * NEWSBT * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_idirlr, sizeof(int) * LR * _nsurf));
	checkCudaErrors(cudaMalloc((void**)&_sgnlr, sizeof(int) * LR * _nsurf));
	checkCudaErrors(cudaMalloc((void**)&_lklr, sizeof(int) * LR * _nsurf));

}

NodalCuda::~NodalCuda()
{
}

void NodalCuda::init()
{
}

void NodalCuda::reset(CrossSection& xs)
{
	checkCudaErrors(cudaMemcpy(_xsnff, &xs.xsnf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsdf, &xs.xsdf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xstf, &xs.xstf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xschif, &xs.chif(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsadf,&xs.xsadf(0, 0), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xssf, &xs.xssf(0, 0, 0), sizeof(XS_PRECISION) * _nxyz * _ng2, cudaMemcpyHostToDevice));

}

void NodalCuda::drive()
{
	::reset << <_blocks, _threads >> > (_hmesh, _xstf, _xsdf, _eta1, _eta2, _m260, _m251, _m253, _m262, _m264, _diagD, _diagDI);
	::calculateTransverseLeakage << <_blocks, _threads >> > (_lktosfc, _neib, _hmesh, _jnet, _trlcff0, _trlcff1, _trlcff2);
	::resetMatrix << <_blocks, _threads >> > (_reigv, _xstf, _xsnff, _xschif, _xssf, _matMs, _matMf, _matM);
	::prepareMatrix << <_blocks, _threads >> > (_m251, _m253, _diagD, _diagDI, _matM, _matMI, _tau, _mu);

	::calculateEven << <_blocks, _threads >> > (_m260, _m262, _m264, _diagD, _diagDI, _matM, _flux,
		_trlcff0, _trlcff2, _dsncff2, _dsncff4, _dsncff6);
	::calculateJnet << <_blocks_sfc, _threads_sfc >> > (_lklr, _idirlr, _sgnlr, _hmesh, _xsadf, _m260, _m262, _m264,
		_diagD, _diagDI, _matM, _matMI, _flux, _trlcff0, _trlcff1,
		_trlcff2, _mu, _tau, _eta1, _eta2, _dsncff2, _dsncff4, _dsncff6, _jnet);
}
