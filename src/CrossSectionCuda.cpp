
#include "CrossSectionCuda.h"

CrossSectionCuda::CrossSectionCuda(const CrossSection& x)
{
	_ng = x.ng();
	_nxyz = x.nxyz();

	checkCudaErrors(cudaMalloc((void**)&_xsnf, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsdf, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xstf, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xskf, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_chif, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xssf, sizeof(XS_PRECISION) * _ng * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsadf, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmacd0, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmaca0, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmacs0, sizeof(XS_PRECISION) * _ng * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmacf0, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmack0, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmacn0, sizeof(XS_PRECISION) * _ng * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmicd, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmica, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmics, sizeof(XS_PRECISION) * _ng * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmicf, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmick, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmicn, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmic2n, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmicd0, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmica0, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmics0, sizeof(XS_PRECISION) * _ng * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmicf0, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmick0, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xsmicn0, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdfmicd, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdfmica, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xddmicd, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xddmica, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdpmicd, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdpmica, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdmmicd, sizeof(XS_PRECISION) * _ng * NPTM * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdmmica, sizeof(XS_PRECISION) * _ng * NPTM * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xddmics, sizeof(XS_PRECISION) * _ng * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdpmics, sizeof(XS_PRECISION) * _ng * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdfmics, sizeof(XS_PRECISION) * _ng * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdmmics, sizeof(XS_PRECISION) * _ng * _ng * NPTM * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdfmicn, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xddmicn, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdpmicn, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdmmicn, sizeof(XS_PRECISION) * _ng * NPTM * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdpmicf, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdfmicf, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xddmicf, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdmmicf, sizeof(XS_PRECISION) * _ng * NPTM * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdpmick, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdfmick, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xddmick, sizeof(XS_PRECISION) * _ng * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_xdmmick, sizeof(XS_PRECISION) * _ng * NPTM * NISO * _nxyz));
	checkCudaErrors(cudaMalloc((void**)&_dpmacd, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dpmaca, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dpmacf, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dpmack, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dpmacn, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dpmacs, sizeof(XS_PRECISION) * _nxyz * _ng * _ng));
	checkCudaErrors(cudaMalloc((void**)&_ddmacd, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_ddmaca, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_ddmacf, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_ddmack, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_ddmacn, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_ddmacs, sizeof(XS_PRECISION) * _nxyz * _ng * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dmmacd, sizeof(XS_PRECISION) * _nxyz * NPTM * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dmmaca, sizeof(XS_PRECISION) * _nxyz * NPTM * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dmmacf, sizeof(XS_PRECISION) * _nxyz * NPTM * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dmmack, sizeof(XS_PRECISION) * _nxyz * NPTM * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dmmacn, sizeof(XS_PRECISION) * _nxyz * NPTM * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dmmacs, sizeof(XS_PRECISION) * _nxyz * NPTM * _ng * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dfmacd, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dfmaca, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dfmacf, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dfmack, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dfmacn, sizeof(XS_PRECISION) * _nxyz * _ng));
	checkCudaErrors(cudaMalloc((void**)&_dfmacs, sizeof(XS_PRECISION) * _nxyz * _ng * _ng));

	checkCudaErrors(cudaMemcpy(_xsnf, x.xsnf(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsdf, x.xsdf(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xstf, x.xstf(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xskf, x.xskf(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_chif, x.chif(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xssf, x.xssf(), sizeof(XS_PRECISION) * _ng * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsadf, x.xsadf(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmacd0, x.xsmacd0(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmaca0, x.xsmaca0(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmacs0, x.xsmacs0(), sizeof(XS_PRECISION) * _ng * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmacf0, x.xsmacf0(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmack0, x.xsmack0(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmacn0, x.xsmacn0(), sizeof(XS_PRECISION) * _ng * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmicd, x.xsmicd(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmica, x.xsmica(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmics, x.xsmics(), sizeof(XS_PRECISION) * _ng * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmicf, x.xsmicf(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmick, x.xsmick(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmicn, x.xsmicn(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmic2n, x.xsmic2n(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmicd0, x.xsmicd0(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmica0, x.xsmica0(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmics0, x.xsmics0(), sizeof(XS_PRECISION) * _ng * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmicf0, x.xsmicf0(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmick0, x.xsmick0(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xsmicn0, x.xsmicn0(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdfmicd, x.xdfmicd(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdfmica, x.xdfmica(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xddmicd, x.xddmicd(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xddmica, x.xddmica(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdpmicd, x.xdpmicd(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdpmica, x.xdpmica(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdmmicd, x.xdmmicd(), sizeof(XS_PRECISION) * _ng * NPTM * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdmmica, x.xdmmica(), sizeof(XS_PRECISION) * _ng * NPTM * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xddmics, x.xddmics(), sizeof(XS_PRECISION) * _ng * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdpmics, x.xdpmics(), sizeof(XS_PRECISION) * _ng * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdfmics, x.xdfmics(), sizeof(XS_PRECISION) * _ng * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdmmics, x.xdmmics(), sizeof(XS_PRECISION) * _ng * _ng * NPTM * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdfmicn, x.xdfmicn(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xddmicn, x.xddmicn(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdpmicn, x.xdpmicn(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdmmicn, x.xdmmicn(), sizeof(XS_PRECISION) * _ng * NPTM * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdpmicf, x.xdpmicf(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdfmicf, x.xdfmicf(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xddmicf, x.xddmicf(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdmmicf, x.xdmmicf(), sizeof(XS_PRECISION) * _ng * NPTM * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdpmick, x.xdpmick(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdfmick, x.xdfmick(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xddmick, x.xddmick(), sizeof(XS_PRECISION) * _ng * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_xdmmick, x.xdmmick(), sizeof(XS_PRECISION) * _ng * NPTM * NISO * _nxyz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dpmacd, x.dpmacd(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dpmaca, x.dpmaca(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dpmacf, x.dpmacf(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dpmack, x.dpmack(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dpmacn, x.dpmacn(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dpmacs, x.dpmacs(), sizeof(XS_PRECISION) * _nxyz * _ng * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_ddmacd, x.ddmacd(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_ddmaca, x.ddmaca(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_ddmacf, x.ddmacf(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_ddmack, x.ddmack(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_ddmacn, x.ddmacn(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_ddmacs, x.ddmacs(), sizeof(XS_PRECISION) * _nxyz * _ng * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmmacd, x.dmmacd(), sizeof(XS_PRECISION) * _nxyz * NPTM * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmmaca, x.dmmaca(), sizeof(XS_PRECISION) * _nxyz * NPTM * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmmacf, x.dmmacf(), sizeof(XS_PRECISION) * _nxyz * NPTM * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmmack, x.dmmack(), sizeof(XS_PRECISION) * _nxyz * NPTM * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmmacn, x.dmmacn(), sizeof(XS_PRECISION) * _nxyz * NPTM * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dmmacs, x.dmmacs(), sizeof(XS_PRECISION) * _nxyz * NPTM * _ng * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dfmacd, x.dfmacd(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dfmaca, x.dfmaca(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dfmacf, x.dfmacf(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dfmack, x.dfmack(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dfmacn, x.dfmacn(), sizeof(XS_PRECISION) * _nxyz * _ng, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_dfmacs, x.dfmacs(), sizeof(XS_PRECISION) * _nxyz * _ng * _ng, cudaMemcpyHostToDevice));
}


__global__ void updateXS(CrossSectionCuda& x, const float* dnst, const float* dppm, const float* dtf, const float* dtm)
{
	int l = threadIdx.x + blockIdx.x * blockDim.x;
	if (l >= x.nxyz()) return;

	x.CrossSection::updateXS(l, dnst, dppm[l], dtf[l], dtm[l]);
}


void CrossSectionCuda::updateXS(const float* dnst, const float* dppm, const float* dtf, const float* dtm)
{
	::updateXS<<<BLOCKS_NODE, THREADS_NODE >>>(*this, dnst, dppm, dtf, dtm);
	cudaDeviceSynchronize();
}


