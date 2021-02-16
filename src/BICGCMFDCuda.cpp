#include "BICGCMFDCuda.h"
#include "JacobiBicgSolverCuda.h"

#define flux(ig, l)   (flux[(l)*_g.ng()+ig])

BICGCMFDCuda::BICGCMFDCuda(Geometry& g, CrossSection& x) : BICGCMFD(g,x)
{
    //_dtil = new CMFD_VAR[_g.nsurf() * _g.ng()]{};
    //_dhat = new CMFD_VAR[_g.nsurf() * _g.ng()]{};
    //_diag = new CMFD_VAR[_g.nxyz() * _g.ng2()]{};
    //_cc = new CMFD_VAR[_g.nxyz() * _g.ng() * NEWSBT]{};
    //_src = new CMFD_VAR[_g.nxyz() * _g.ng()]{};
    //_psi = new CMFD_VAR[_g.nxyz()]{};
    //_epsl2 = 1.E-5;
}

BICGCMFDCuda::~BICGCMFDCuda()
{
}

void BICGCMFDCuda::init()
{
    _epsbicg = 1.E-4;
    _nmaxbicg = 10;

    _eshift = 0.0;
    iter = 0;

    _ls = new JacobiBicgSolverCuda(_g);

    checkCudaErrors(cudaMalloc((void**)&_gammad_dev, sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&_gamman_dev, sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&_errl2_dev, sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&_dtil, sizeof(CMFD_VAR) * _g.nsurf() * _g.ng()));
    checkCudaErrors(cudaMalloc((void**)&_dhat, sizeof(CMFD_VAR) * _g.nsurf() * _g.ng()));
    checkCudaErrors(cudaMalloc((void**)&_diag, sizeof(CMFD_VAR) * _g.nxyz() * _g.ng2()));
    checkCudaErrors(cudaMalloc((void**)&_unshifted_diag, sizeof(CMFD_VAR) * _g.nxyz() * _g.ng2()));
    checkCudaErrors(cudaMalloc((void**)&_cc, sizeof(CMFD_VAR) * _g.ngxyz() * NEWSBT));
    checkCudaErrors(cudaMalloc((void**)&_src, sizeof(CMFD_VAR) * _g.ngxyz()));
    checkCudaErrors(cudaMalloc((void**)&_psi, sizeof(CMFD_VAR) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_psid, sizeof(CMFD_VAR) * _g.nxyz()));
    cudaDeviceSynchronize();
}

__global__ void upddtil(BICGCMFDCuda& self) {
    int ls = threadIdx.x + blockIdx.x * blockDim.x;
    if (ls >= self.g().nsurf()) return;

    self.CMFD::upddtil(ls);
}

void BICGCMFDCuda::upddtil() {
    ::upddtil << <BLOCKS_SURFACE, THREADS_SURFACE >> > (*this);
    cudaDeviceSynchronize();
}

__global__ void upddhat(BICGCMFDCuda& self, SOL_VAR* flux, SOL_VAR* jnet) {
    int ls = threadIdx.x + blockIdx.x * blockDim.x;
    if (ls >= self.g().nsurf()) return;

    self.CMFD::upddhat(ls, flux, jnet);
}

void BICGCMFDCuda::upddhat(SOL_VAR* flux, SOL_VAR* jnet) {
    ::upddhat << <BLOCKS_SURFACE, THREADS_SURFACE >> > (*this, flux, jnet);
    cudaDeviceSynchronize();
}

__global__ void setls(BICGCMFDCuda& self, const double& eigv, const double& reigvs) {

    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    self.BICGCMFD::setls(l);
    self.BICGCMFD::updls(l, reigvs);
}

void BICGCMFDCuda::setls(const double& eigv) {
    double reigvs = 0.0;
    if (eshift() != 0.0) reigvs = 1. / (eigv + eshift());
    ::setls<<<BLOCKS_NODE, THREADS_NODE>>>(*this, eigv, reigvs);
    cudaDeviceSynchronize();
    
    _ls->facilu(_diag, _cc);
}

__global__ void updls(BICGCMFDCuda& self, const double& reigvs) {

    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    self.BICGCMFD::updls(l, reigvs);
}


void BICGCMFDCuda::updls(const double& reigvs) {
    ::updls << <BLOCKS_NODE, THREADS_NODE >> > (*this, reigvs);
    cudaDeviceSynchronize();
}


__global__ void updjnet(BICGCMFDCuda& self, SOL_VAR* flux, SOL_VAR* jnet) {
    int ls = threadIdx.x + blockIdx.x * blockDim.x;
    if (ls >= self.g().nsurf()) return;

    self.CMFD::updjnet(ls, flux, jnet);
}


void BICGCMFDCuda::updjnet(SOL_VAR* flux, SOL_VAR* jnet)
{
    ::updjnet << <BLOCKS_SURFACE, THREADS_SURFACE >> > (*this, flux, jnet);
    cudaDeviceSynchronize();
}

__global__ void updpsi(BICGCMFDCuda& self, const SOL_VAR* flux)
{
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    self.psid(l) = self.psi(l);
    self.CMFD::updpsi(l, flux);
}

void BICGCMFDCuda::updpsi(const SOL_VAR* flux)
{
    ::updpsi << <BLOCKS_NODE, THREADS_NODE >> > (*this, flux);
    cudaDeviceSynchronize();
}


__global__ void axb(BICGCMFDCuda& self, SOL_VAR* flux, SOL_VAR* aflux) {

    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    for (int ig = 0; ig < self.g().ng(); ++ig) {
        aflux[l*self.g().ng() + ig] = self.CMFD::axb(ig, l, flux);
    }
}

void BICGCMFDCuda::axb(SOL_VAR* flux, SOL_VAR* aflux)
{
    ::axb << <BLOCKS_NODE, THREADS_NODE >> > (*this, flux, aflux);
    cudaDeviceSynchronize();
}


__global__ void updsrc(BICGCMFDCuda& self, const double& reigvdel) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    double fs = self.psi(l) * reigvdel;
    for (int ig = 0; ig < self.g().ng(); ++ig) {
        self.src(ig, l) = self.x().chif(ig, l) * fs;
    }

}

void BICGCMFDCuda::updsrc(const double& reigvdel) {
    ::updsrc << <BLOCKS_NODE, THREADS_NODE >> > (*this, reigvdel);
    cudaDeviceSynchronize();
}

__global__ void psierr(int nxyz, const CMFD_VAR* psid, const CMFD_VAR* psi, float* errl2, double* gammad, double* gamman) {
    __shared__ float errl2_cache[NTHREADSPERBLOCK];
    __shared__ double gammad_cache[NTHREADSPERBLOCK];
    __shared__ double gamman_cache[NTHREADSPERBLOCK];

    int l = blockIdx.x * blockDim.x + threadIdx.x;
    errl2_cache[threadIdx.x] = 0.0;
    gammad_cache[threadIdx.x] = 0.0;
    gamman_cache[threadIdx.x] = 0.0;

    while (l < nxyz) {
        CMFD_VAR err = psi[l] - psid[l];
        errl2_cache[threadIdx.x] += err*  err;
        gammad_cache[threadIdx.x] += psid[l] * psi[l];
        gamman_cache[threadIdx.x] += psi[l] * psi[l];
        l += gridDim.x * blockDim.x;
    }
    __syncthreads();  // required because later on the current thread is
                      // accessing data written by another thread
    
    l = NTHREADSPERBLOCK / 2;
    while (l > 0) {
        if (threadIdx.x < l) {
            errl2_cache[threadIdx.x] += errl2_cache[threadIdx.x + l];
            gammad_cache[threadIdx.x] += gammad_cache[threadIdx.x + l];
            gamman_cache[threadIdx.x] += gamman_cache[threadIdx.x + l];
        }
        __syncthreads();
        l /= 2;  // not sure bitwise operations are actually faster
    }

    if (threadIdx.x == 0) {
        atomicAdd(errl2, errl2_cache[0]);
        atomicAdd(gammad, gammad_cache[0]);
        atomicAdd(gamman, gamman_cache[0]);
    }
}

void BICGCMFDCuda::wiel(const int& icy, const SOL_VAR* flux, double& reigvs, double& eigv, double& reigv, float& errl2) {


    updpsi(flux);

    psierr << <BLOCKS_NODE, THREADS_NODE >> > (g().nxyz(), _psid, _psi, _errl2_dev, _gammad_dev, _gamman_dev);
    cudaDeviceSynchronize();

    double gamman = 0;
    double gammad = 0;
    errl2 = 0;
    cudaMemcpy(&gammad, _gammad_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gamman, _gamman_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&errl2, _errl2_dev, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    //compute new eigenvalue
    double eigvd = eigv;
    double gamma = gammad / gamman;
    eigv = 1 / (reigv * gamma + (1 - gamma) * reigvs);
    reigv = 1 / eigv;

    errl2 = sqrt(errl2 / gammad);
    double erreig = abs(eigv - eigvd);;

    double eigvs = eigv;
    if (icy >= 0) {
        eigvs += _eshift;
    }

    reigvs = 0;
    if (_eshift != 0.0) reigvs = 1 / eigvs;

}



void BICGCMFDCuda::drive(double& eigv, SOL_VAR* flux, float& errl2) {

    int icmfd = 0;
    double reigv = 1. / eigv;
    double reigvs = 0.0;

    if (_eshift != 0.0) reigvs = 1. / (eigv + _eshift);

    for (int iout = 0; iout < _ncmfd; ++iout) {
        ++iter; ++icmfd;
        double reigvdel = reigv - reigvs;
        updsrc(reigvdel);

        float r20 = 0.0;
        _ls->reset(_diag, _cc, flux, _src, r20);

        double r2 = 0.0;
        for (int iin = 0; iin < _nmaxbicg; ++iin) {
            //solve linear system A*phi = src
            _ls->solve(_diag, _cc, r20, flux, r2);
            if (r2 < _epsbicg) break;
        }

        //wielandt shift
        wiel(iter, flux, reigvs, eigv, reigv, errl2);

        if (reigvs != 0.0) updls(reigvs);

        int negative = 0;
        for (int l = 0; l < _g.nxyz(); ++l) {
            for (int ig = 0; ig < _g.ng(); ++ig) {
                if (flux(ig, l) < 0) {
                    ++negative;
                }
            }
        }
        if (negative == _g.ngxyz()) {
            negative = 0;
        }

        if (negative != 0 && icmfd < 5 * _ncmfd) iout--;

        printf("IOUT : %d, EIGV : %9.7f , ERRL2 : %12.5E, NEGATIVE : %d\n", iter, eigv, errl2, negative);

        if (errl2 < _epsl2) break;

    }
}