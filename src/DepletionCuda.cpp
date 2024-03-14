#include "DepletionCuda.h"

#define xsmica(ig, iiso, l) xsmica[(l) * self.g().ng() * NISO + (iiso) * self.g().ng() + (ig)]
#define xsmicf(ig, iiso, l) xsmicf[(l) * self.g().ng() * NISO + (iiso) * self.g().ng() + (ig)]
#define xsmic2n(ig, l) xsmic2n[(l) * self.g().ng() + (ig)]
#define flux(ig, l) flux[(l) * self.g().ng() + (ig)]
#define ati(iiso, l) ati[(l) * NISO + (iiso)]
#define atd(iiso, l) atd[(l) * NISO + (iiso)]
#define atavg(iiso, l) atavg[(l) * NISO + (iiso)]

DepletionCuda::DepletionCuda(GeometryCuda& g) : Depletion(g)
{
}

DepletionCuda::~DepletionCuda()
{
}


__global__ void init(int* _nheavy, int* _ihvys, int* _hvyids, int* _reactype, int* _hvyupd, float* _dcy) {
	_nheavy[0] = 10; _ihvys[0] = 0;
	_nheavy[1] = 7;	 _ihvys[1] = 10;
	_nheavy[2] = 8;	 _ihvys[2] = 17;
	_nheavy[3] = 2;	 _ihvys[3] = 25;

	int idx = -1;
	++idx; _hvyids[idx] = U234; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = U235; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = U236; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = NP37; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = PU48; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = PU49; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = PU40; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = PU41; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = PU42; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = AM43; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = U238; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = NP39; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = PU49; _reactype[idx] = R_DEC; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = PU40; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = PU41; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = PU42; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = AM43; _reactype[idx] = R_CAP; _hvyupd[idx] = UPDATE;
	++idx; _hvyids[idx] = U238; _reactype[idx] = R_CAP; _hvyupd[idx] = HOLD;
	++idx; _hvyids[idx] = NP37; _reactype[idx] = R_N2N; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = PU48; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = PU49; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = PU40; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = PU41; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = PU42; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = AM43; _reactype[idx] = R_CAP; _hvyupd[idx] = LEAVE;
	++idx; _hvyids[idx] = PU48; _reactype[idx] = R_CAP; _hvyupd[idx] = HOLD;
	++idx; _hvyids[idx] = U234; _reactype[idx] = R_DEC; _hvyupd[idx] = LEAVE;


	for (int idcy = 0; idcy < NDCY; ++idcy) {
		_dcy[ISODCY[idcy]] = DCY[idcy];
	}

}
void DepletionCuda::init()
{
	_nhvychn = 4;
	_nhvyids = 10 + 7 + 8 + 2;


	checkCudaErrors(cudaMalloc((void**)&_nheavy, sizeof(int) * _nhvychn));
	checkCudaErrors(cudaMalloc((void**)&_ihvys, sizeof(int) * _nhvychn));
	checkCudaErrors(cudaMalloc((void**)&_hvyids, sizeof(int) * _nhvyids));
	checkCudaErrors(cudaMalloc((void**)&_reactype, sizeof(int) * _nhvyids));
	checkCudaErrors(cudaMalloc((void**)&_hvyupd, sizeof(int) * _nhvyids));
	checkCudaErrors(cudaMalloc((void**)&_dcy, sizeof(float) * NDEP));
	checkCudaErrors(cudaMalloc((void**)&_cap, sizeof(float) * NDEP * _g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_rem, sizeof(float) * NDEP * _g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_fis, sizeof(float) * NDEP * _g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_tn2n, sizeof(float) * _g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_dnst, sizeof(float) * NISO * _g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_dnst_new, sizeof(float) * NISO * _g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_dnst_avg, sizeof(float) * NISO * _g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_burn, sizeof(float) * _g.nxyz()));
	checkCudaErrors(cudaMalloc((void**)&_h2on, sizeof(float) * _g.nxyz()));

	::init << <1, 1 >> > (_nheavy, _ihvys, _hvyids, _reactype, _hvyupd, _dcy);
	checkCudaErrors(cudaDeviceSynchronize());

	_b10ap = 19.8;
	_b10fac = _b10ap / (_b10ap * B10AW + (100. - _b10ap) * B11AW);
	_b10wp = 100. * B10AW * _b10fac;
}

__global__ void dep(DepletionCuda& self, float tsec)
{
	int l = threadIdx.x + blockIdx.x * blockDim.x;
	if (l >= self.g().nxyz()) return;

	self.Depletion::dep(l, tsec, self.dnst(), self.dnst_new(), self.dnst_avg());

}

__global__ void eqxe(DepletionCuda& self, const float* xsmica, const float* xsmicf, const double* flux, float fnorm)
{

	if (self.xeopt() != XEType::XE_EQ) return;

	int l = threadIdx.x + blockIdx.x * blockDim.x;
	if (l >= self.g().nxyz()) return;

	if (xsmicf(1, U235, l) == 0) return;

	self.Depletion::eqxe(l, xsmica, xsmicf, flux, fnorm);
}

__global__ void pickData(DepletionCuda& self, const float* xsmica, const float* xsmicf, const float* xsmic2n, const double* flux, float fnorm) {

	int l = threadIdx.x + blockIdx.x * blockDim.x;
	if (l >= self.g().nxyz()) return;

	if (xsmica(1, U235, l) == 0) return;

	self.Depletion::pickData(l, xsmica, xsmicf, xsmic2n, flux, fnorm);
}

__global__ void updateH2ODensity(DepletionCuda& self, const float* dm, float ppm) {
	int l = threadIdx.x + blockIdx.x * blockDim.x;
	if (l >= self.g().nxyz()) return;

	self.Depletion::updateH2ODensity(l, dm, ppm);
}


void DepletionCuda::dep(const float& tsec)
{
	::dep << <BLOCKS_NODE, THREADS_NODE >> > (*this, tsec);
	checkCudaErrors(cudaDeviceSynchronize());
}

void DepletionCuda::eqxe(const float* xsmica, const float* xsmicf, const double* flux, const float& fnorm)
{

	if (xeopt() != XEType::XE_EQ) return;

	::eqxe << <BLOCKS_NODE, THREADS_NODE >> > (*this, xsmica, xsmicf, flux, fnorm);
	checkCudaErrors(cudaDeviceSynchronize());
}

void DepletionCuda::pickData(const float* xsmica, const float* xsmicf, const float* xsmic2n, const double* flux, const float& fnorm) {

	::pickData << <BLOCKS_NODE, THREADS_NODE >> > (*this, xsmica, xsmicf, xsmic2n, flux, fnorm);
	checkCudaErrors(cudaDeviceSynchronize());
}

void DepletionCuda::updateH2ODensity(const float* dm, const float& ppm) {
	::updateH2ODensity << <BLOCKS_NODE, THREADS_NODE >> > (*this, dm, ppm);
	checkCudaErrors(cudaDeviceSynchronize());
}

void DepletionCuda::setDensity(const float* dnst)
{
	checkCudaErrors(cudaMemcpy(_dnst, dnst, sizeof(float) * NISO * _g.nxyz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
}

void DepletionCuda::setBurnup(const float* burn)
{
	checkCudaErrors(cudaMemcpy(_burn, burn, sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
}

void DepletionCuda::setH2ORatio(const float* h2on)
{
	checkCudaErrors(cudaMemcpy(_h2on, h2on, sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
}
