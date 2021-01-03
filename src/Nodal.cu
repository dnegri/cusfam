#include <cuda_runtime.h>
#include <math.h>
#include "Nodal.h"
#include "const.cuh"

#define d_jnet(lr,idir,lkg)	(d_jnet[lkg*NEWSBT + idir*LR + lr])
#define d_trlcff0(idir,lkg)	(d_trlcff0[lkg*NDIRMAX + idir])
#define d_trlcff1(idir,lkg)	(d_trlcff1[lkg*NDIRMAX + idir])
#define d_trlcff2(idir,lkg)	(d_trlcff2[lkg*NDIRMAX + idir])
#define d_hmesh(idir,lk)		(d_hmesh[lk*NDIRMAX + idir])
#define d_neib(lr,idir,lk)	(d_neib[lk*NEWSBT + idir*LR + lr])
#define d_lkg3(ig,l,k)		(k*(d_nxy*d_ng)+l*d_ng+ig)
#define d_lkg2(ig,lk)			(lk*d_ng+ig)

#define d_eta1(idir,lkg)	(d_eta1[lkg*NDIRMAX + idir])
#define d_eta2(idir,lkg)	(d_eta2[lkg*NDIRMAX + idir])
#define d_m260(idir,lkg)	(d_m260[lkg*NDIRMAX + idir])
#define d_m251(idir,lkg)	(d_m251[lkg*NDIRMAX + idir])
#define d_m253(idir,lkg)	(d_m253[lkg*NDIRMAX + idir])
#define d_m262(idir,lkg)	(d_m262[lkg*NDIRMAX + idir])
#define d_m264(idir,lkg)	(d_m264[lkg*NDIRMAX + idir])
#define d_diagD(idir,lkg) (d_diagD[lkg*NDIRMAX + idir])
#define d_diagDI(idir,lkg)	(d_diagDI[lkg*NDIRMAX + idir])
#define d_mu(i,j,lkdir)	(d_mu(lkdir*d_ng2 + j*d_ng + i))
#define t_dau(i,j,lkdir)	(d_mu(lkdir*d_ng2 + j*d_ng + i))



__global__ void reset(float * d_xstf, float * d_xsdf, float* d_eta1, float* d_eta2, float* d_m260,float* d_m251,float* d_m253,float* d_m262,float* d_m264, float* d_diagD, float* d_diagDI) {
	int lkg = threadIdx.x + blockIdx.x * blockDim.x;
	int lk = lkg * d_rng;

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
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
		d_eta1(idir, lkg) = (kp * coshkp + bfcff1 + 6 * bfcff3) * oddtemp;
		d_eta2(idir, lkg) = (kp * sinhkp + 3 * bfcff2 + 10 * bfcff4) * eventemp;

		//set to variables that depends on node properties by integrating of Pi* pj over - 1 ~1
		d_m260(idir, lkg) = 2 * d_eta2(idir, lkg);
		d_m251(idir, lkg) = 2 * (kp * coshkp - sinhkp + 5 * bfcff3) * oddtemp;
		d_m253(idir, lkg) = 2 * (kp * (15 + kp2) * coshkp - 3 * (5 + 2 * kp2) * sinhkp) * oddtemp * rkp2;
		d_m262(idir, lkg) = 2 * (-3 * kp * coshkp + (3 + kp2) * sinhkp + 7 * kp * bfcff4) * eventemp * rkp;
		d_m264(idir, lkg) = 2 * (-5 * kp * (21 + 2 * kp2) * coshkp + (105 + 45 * kp2 + kp4) * sinhkp) * eventemp * rkp3;
		if (d_m264(idir, lkg) == 0.0) d_m264(idir, lkg) = 1.e-10;

		d_diagD(idir, lkg) = 4 * d_xsdf[lkg] / (d_hmesh(idir, lk) * d_hmesh(idir, lk));
		d_diagDI(idir, lkg) = 1.0 / d_diagD(idir, lkg);
	}

}

__global__ void resetMatrix(double & d_reigv, float* d_xstf, float* d_xsnff, float* d_xschif, float* d_xssf, float* d_matMs, float* d_matMf, float* d_matM) {
	int lkgd = threadIdx.x + blockIdx.x * blockDim.x;
	int lk  = lkgd * d_rng;
	int igd = lkgd % d_ng;
	int lkgg = lkgd * d_ng;

	for (size_t igs = 0; igs < d_ng; igs++)
	{
		d_matMs[lkgg+igs] = 0.0;
	}

	d_matMs[lkgg + igd] = d_xstf[lkgd];

	for (size_t igs = 0; igs < d_ng; igs++)
	{
		auto lkggs = lkgg + igs;
		d_matMs[lkggs] = d_matMs[lkggs] - d_xssf[lkggs];
		int lkgs = d_lkg2(igs, lk);
		d_matMf[lkggs] = d_xschif[lkgd] * d_xsnff[lkgs];

		d_matM[lkggs] = d_matMs[lkggs] - d_reigv * d_matMf[lkggs];
	}
}

__global__ void inverseMatM(float* d_matM, float* d_matMI) {
	int lk = threadIdx.x + blockIdx.x * blockDim.x;
	int lkg2 = lk * d_ng2;

	auto det = d_matM[lkg2+0] * d_matM[lkg2+3] - d_matM[1, lk] * d_matM[2, lk];

	if (abs(det) < 1.E-10) {
		auto rdet = 1 / det;
		d_matMI[lkg2 + 0] = rdet * d_matM[lkg2 + 3];
		d_matMI[lkg2 + 1] = -rdet * d_matM[lkg2 + 1];
		d_matMI[lkg2 + 2] = -rdet * d_matM[lkg2 + 2];
		d_matMI[lkg2 + 3] = rdet * d_matM[lkg2 + 0];
	} else {
		d_matMI[lkg2 + 0] = 0;
		d_matMI[lkg2 + 1] = 0;
		d_matMI[lkg2 + 2] = 0;
		d_matMI[lkg2 + 3] = 0;
	}

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		for (size_t igd = 0; igd < d_ng; igd++)
		{
			auto lkg  = lk*d_ng + igd;
			auto lkg2 = lk * d_ng2 + igd*d_ng;
			auto tau1 = m033 * (d_diagDI(idir, lkg) / d_m253(idir, lkg));

			for (size_t igs = 0; igs < d_ng; igs++)
			{
				d_tau(igs, igd, lkdir) = tau1 * d_matM[lkg+ igs];

                // mu=m011_inv*M_inv*D*(m231*I+m251*tau)
                tempz(igs,1)=d_m251(1,l,k,idir)*tau(igs,1,idir,l,k)
                tempz(igs,2)=d_m251(2,l,k,idir)*tau(igs,2,idir,l,k)
                tempz(1,1)=tempz(1,1)+m231
                tempz(2,2)=tempz(2,2)+m231
                tempz(igs,1)=diagD(1,l,k,idir)*tempz(igs,1)
                tempz(igs,2)=diagD(2,l,k,idir)*tempz(igs,2)

			}

		}
	}

}



__device__ void trlcffbyintg(float* avgtrl3, float * hmesh3, float* trlcff1, float* trlcff2) {
	float sh[4];

	auto rh = (1 / ((hmesh3[1] + hmesh3[0] + hmesh3[2]) * (hmesh3[1] + hmesh3[0]) * (hmesh3[0] + hmesh3[2])));
	sh[1] = (2 * hmesh3[1] + hmesh3[0]) * (hmesh3[1] + hmesh3[0]);
	sh[2] = hmesh3[1] + hmesh3[0];
	sh[3] = (hmesh3[0] + 2 * hmesh3[2]) * (hmesh3[0] + hmesh3[2]);
	sh[4] = hmesh3[0] + hmesh3[2];

	if (hmesh3[LEFT] == 0.0) {
		*trlcff1 = 0.125 * (5. * avgtrl3[CENTER] + avgtrl3[RIGHT]);
		*trlcff2 = 0.125 * (-3. * avgtrl3[CENTER] + avgtrl3[RIGHT]);
	}
	else if (hmesh3[RIGHT] == 0.0) {
		*trlcff1 = -0.125 * (5. * avgtrl3[CENTER] + avgtrl3[LEFT]);
		*trlcff2 = 0.125 * (-3. * avgtrl3[CENTER] + avgtrl3[LEFT]);
	}
	else {
		*trlcff1 = 0.5 * rh * hmesh3[0] * ((avgtrl3[CENTER] - avgtrl3[LEFT]) * sh[3] + (avgtrl3[RIGHT] - avgtrl3[CENTER]) * sh[1]);
		*trlcff2 = 0.5 * rh * (hmesh3[0] * hmesh3[0]) * ((avgtrl3[LEFT] - avgtrl3[CENTER]) * sh[4] + (avgtrl3[RIGHT] - avgtrl3[CENTER]) * sh[2]);
	}

}

__global__ void calculateTransverseLeakage(float * d_jnet, float * d_trlcff0)
{
	int lkg = threadIdx.x + blockIdx.x * blockDim.x;
	int lk = lkg * d_rng;

	float avgjnet[NDIRMAX];

	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		avgjnet[idir] = (d_jnet(RIGHT, idir, lkg) - d_jnet(LEFT, idir, lkg)) * d_hmesh(idir, lk);
	}

	d_trlcff0(XDIR,lkg) = avgjnet[YDIR] + avgjnet[ZDIR];
	d_trlcff0(YDIR,lkg) = avgjnet[XDIR] + avgjnet[ZDIR];
	d_trlcff0(ZDIR,lkg) = avgjnet[XDIR] + avgjnet[YDIR];
}

__global__ void calculateTransverseLeakage2(float* d_trlcff0, float* d_trlcff1, float* d_trlcff2)
{
	int lkg = threadIdx.x + blockIdx.x * blockDim.x;
	int lk = lkg * d_rng;
	int k = lk / d_nxy;
	int ig = lkg % d_ng;


	for (size_t idir = 0; idir < NDIRMAX; idir++)
	{
		float avgtrl3[LRC] = { 0.0 }, hmesh3[LRC] = { 0.0 };

		avgtrl3[CENTER] = d_trlcff0(idir, lkg);
		hmesh3[CENTER] = d_hmesh(idir, lk);

		for (size_t lr = 0; lr < LR; lr++)
		{
			int ln = d_neib(lr, idir, lk);
			int lnkg = d_lkg3(ig, ln, k);
			int lnk = lnkg * d_rng;
			avgtrl3[lr] = d_trlcff0(idir, lnkg);
			hmesh3[lr] = d_hmesh(idir, lnk);
		}

		trlcffbyintg(avgtrl3, hmesh3, &d_trlcff1(idir, lkg), &d_trlcff2(idir, lkg));

	}
}

Nodal::~Nodal()
{
}

Nodal::Nodal()
{
	_blocks  = dim3(_ng * _nxy * _nz / NTHREADSPERBLOCK + 1, 1, 1);
	_threads = dim3(NTHREADSPERBLOCK, 1, 1);

}

void Nodal::reset()
{

	::reset << <_blocks, _threads >> > (d_xstf, d_xsdf, d_eta1, d_eta2,
										d_m260,d_m251,d_m253,d_m262,d_m264,d_diagD, d_diagDI);

}


void Nodal::drive()
{
	reset();
	calculateTransverseLeakage();
}

void Nodal::calculateTransverseLeakage()
{
	::calculateTransverseLeakage << <_blocks, _threads >> > (d_jnet, d_trlcff0);
}


