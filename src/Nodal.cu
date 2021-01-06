#include <cuda_runtime.h>
#include <math.h>
#include "Nodal.h"
#include "const.cuh"

static const int m011 = 2. / 3.;
static const int m022 = 2. / 5.;
static const int m033 = 2. / 7.;
static const int m044 = 2. / 9.;
static const int m220 = 6.;
static const int rm220 = 1 / 6.;
static const int m240 = 20.;
static const int m231 = 10.;
static const int m242 = 14.;


#define d_neib(lr,idir,lk)	(d_neib[lk*NEWSBT + idir*LR + lr])
#define d_jnet(ig,lr,idir,lk)	(d_jnet[(lk*NEWSBT + idir*LR + lr)*d_ng + ig])
#define d_trlcff0(ig,idir,lk)	(d_trlcff0[(lk*NDIRMAX + idir)*d_ng + ig])
#define d_trlcff1(ig,idir,lk)	(d_trlcff1[(lk*NDIRMAX + idir)*d_ng + ig])
#define d_trlcff2(ig,idir,lk)	(d_trlcff2[(lk*NDIRMAX + idir)*d_ng + ig])
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
#define d_flux(ig,lk)	(d_flux[lk*d_ng+ig])


__global__ void reset(float* d_xstf, float* d_xsdf, float* d_eta1, float* d_eta2, float* d_m260, float* d_m251, float* d_m253, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI) {
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

__global__ void resetMatrix(double& d_reigv, float* d_xstf, float* d_xsnff, float* d_xschif, float* d_xssf, float* d_matMs, float* d_matMf, float* d_matM) {
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
	int lkg0 = lk * d_ng;

	auto det = d_matM(0, 0, lk) * d_matM(1, 1, lk) - d_matM(1, 0, lk) * d_matM(0, 1, lk);

	if (abs(det) < 1.E-10) {
		auto rdet = 1 / det;
		d_matMI(0, 0, lk) =  rdet * d_matM(1, 1, lk);
		d_matMI(1, 0, lk) = -rdet * d_matM(1, 0, lk);
		d_matMI(0, 1, lk) = -rdet * d_matM(0, 1, lk);
		d_matMI(1, 1, lk) =  rdet * d_matM(0, 0, lk);
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

		float tempz[2][2] = {0.0};

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

__global__ void calculateTransverseLeakage(float* d_jnet, float* d_trlcff0, float* d_trlcff1, float* d_trlcff2)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;

	float avgjnet[NDIRMAX];

	for (size_t ig = 0; ig < d_ng; ig++)
	{
		for (size_t idir = 0; idir < NDIRMAX; idir++)
		{
			avgjnet[idir] = (d_jnet(ig, RIGHT, idir, lk) - d_jnet(ig, LEFT, idir, lk)) * d_hmesh(idir, lk);
		}

		d_trlcff0(ig, XDIR, lk) = avgjnet[YDIR] + avgjnet[ZDIR];
		d_trlcff0(ig, YDIR, lk) = avgjnet[XDIR] + avgjnet[ZDIR];
		d_trlcff0(ig, ZDIR, lk) = avgjnet[XDIR] + avgjnet[YDIR];
	}

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		float avgtrl3[LRC] = { 0.0 }, hmesh3[LRC] = { 0.0 };

		hmesh3[CENTER] = d_hmesh(idir, lk);

		for (size_t lr = 0; lr < LR; lr++)
		{
			auto lnk = d_neib(lr, idir, lk);
			hmesh3[lr] = d_hmesh(idir, lnk);

			for (size_t ig = 0; ig < d_ng; ig++)
			{
				avgtrl3[CENTER] = d_trlcff0(ig, idir, lk);
				avgtrl3[lr] = d_trlcff0(ig, idir, lnk);
				trlcffbyintg(avgtrl3, hmesh3, d_trlcff1(ig, idir, lk), d_trlcff2(ig, idir, lk));
			}
		}
	}
}

__global__ void calculateEven(float* d_m260, float* d_m262, float* d_m264, float* d_diagD, float* d_diagDI, float* d_matM, double* d_flux, float* d_trlcff0, float* d_trlcff1, float* d_trlcff2)
{
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	int lkd0 = lk * NDIRMAX;

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		auto lkd = lkd0 + idir;
		float at2[2][2], a[2][2], rm4464[2], bt1[2], bt2[2];

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
			bt2[igd] = 2 * (d_matM(0, igd, lk) * d_flux(0, lk) + d_matM(1, igd, lk) * d_flux(0, lk) + d_trlcff0(igd, idir, lk));
			bt1[igd] = m022 * rm220 * d_diagDI(igd, lkd) * bt2[igd];
		}

		for (size_t igd = 0; igd < d_ng; igd++)
		{
			auto mu1 = rm4464[igd] * d_m262(igd, lkd);
			for (size_t igs = 0; igs < d_ng; igs++)
			{
			}
		}
	}
}
Nodal::~Nodal()
{
}

Nodal::Nodal()
{
	_blocks = dim3(_ng * _nxy * _nz / NTHREADSPERBLOCK + 1, 1, 1);
	_threads = dim3(NTHREADSPERBLOCK, 1, 1);

}

void Nodal::reset()
{

	::reset << <_blocks, _threads >> > (d_xstf, d_xsdf, d_eta1, d_eta2,
		d_m260, d_m251, d_m253, d_m262, d_m264, d_diagD, d_diagDI);

}


void Nodal::drive()
{
	::reset << <_blocks, _threads >> > (d_xstf, d_xsdf, d_eta1, d_eta2, d_m260, d_m251, d_m253, d_m262, d_m264, d_diagD, d_diagDI);
	::calculateTransverseLeakage << <_blocks, _threads >> > (d_jnet, d_trlcff0,  d_trlcff1, d_trlcff2);
	::resetMatrix << <_blocks, _threads >> > (*d_reigv, d_xstf, d_xsnff, d_xschif, d_xssf, d_matMs, d_matMf, d_matM);
	::prepareMatrix << <_blocks, _threads >> > (d_m251, d_m253, d_diagD, d_diagDI, d_matM, d_matMI, d_tau, d_mu);
}

