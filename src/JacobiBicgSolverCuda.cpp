//
// Created by JOO IL YOON on 2021/01/30.
//

#include "JacobiBicgSolverCuda.h"
#include "mat2g.h"
#include "myblascuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define diag(igs, ige, l) diag[l*_g->ng2()+ige*_g->ng()+igs]
#define cc(lr, idir, ig, l) cc[l*_g->ng()*NDIRMAX*LR+ig*NDIRMAX*LR+idir*LR+lr]
#define src(ig, l) src[l*_g->ng()+ig]
#define aphi(ig, l) aphi[l*_g->ng()+ig]
#define b(ig, l) b[l*_g->ng()+ig]
#define x(ig, l) x[l*_g->ng()+ig]
#define flux(ig, l) flux[l*_g->ng()+ig]
#define b1d(ig, l)   b1d[(l*_g->ng())+ig]
#define x1d(ig, l)   x1d[(l*_g->ng())+ig]
#define b01d(ig, l)  _b01d[(l*_g->ng())+ig]
#define s1dl(ig, l)  _s1dl[(l*_g->ng())+ig]
#define b03d(ig, l)  _b03d[(l*_g->ng())+ig]
#define s3d(ig, l)  _s3d[(l*_g->ng())+ig]
#define s3dd(ig, l)  _s3dd[(l*_g->ng())+ig]


#define vr(ig, l)   _vr[(l*_g->ng())+ig]
#define vr0(ig, l)  _vr0[(l*_g->ng())+ig]
#define vp(ig, l)   _vp[(l*_g->ng())+ig]
#define vv(ig, l)   _vv[(l*_g->ng())+ig]
#define vs(ig, l)   _vs[(l*_g->ng())+ig]
#define vt(ig, l)   _vt[(l*_g->ng())+ig]
#define vy(ig, l)   _vy[(l*_g->ng())+ig]
#define vz(ig, l)   _vz[(l*_g->ng())+ig]
#define y1d(ig, l)   _y1d[(l*_g->ng())+ig]
#define b1i(ig, l)   _b1i[(l*_g->ng())+ig]



JacobiBicgSolverCuda::JacobiBicgSolverCuda(Geometry& g) {

    _g = &g;

    checkCudaErrors(cudaMalloc((void**)&_r20_dev, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&_r2_dev, sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)& _crho_dev, sizeof(CMFD_VAR)));
    checkCudaErrors(cudaMalloc((void**)&_r0v_dev, sizeof(CMFD_VAR)));
    checkCudaErrors(cudaMalloc((void**)&_pts_dev, sizeof(CMFD_VAR)));
    checkCudaErrors(cudaMalloc((void**)&_ptt_dev, sizeof(CMFD_VAR)));
    checkCudaErrors(cudaMalloc((void**)&_vz, sizeof(SOL_VAR) * _g->ngxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vy, sizeof(SOL_VAR) * _g->ngxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vr, sizeof(CMFD_VAR) * _g->ngxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vr0, sizeof(CMFD_VAR) * _g->ngxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vp, sizeof(CMFD_VAR) * _g->ngxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vv, sizeof(CMFD_VAR) * _g->ngxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vs, sizeof(CMFD_VAR) * _g->ngxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vt, sizeof(CMFD_VAR) * _g->ngxyz()));
    checkCudaErrors(cudaMalloc((void**)&_delinv, sizeof(CMFD_VAR) *_g->ng2() * _g->nxyz()));


}

JacobiBicgSolverCuda::~JacobiBicgSolverCuda() {

}

__global__ void reset(JacobiBicgSolverCuda& self, CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux, CMFD_VAR* src, float& r20) {

    __shared__ float r2[NTHREADSPERBLOCK];

    int tid = threadIdx.x;
    int l = tid + blockIdx.x * blockDim.x;

    if (l >= self.g().nxyz()) return;

    r2[tid] = self.JacobiBicgSolver::reset(l, diag, cc, flux, src);

    __syncthreads();

    //printf("r2[tid] : %d %e\n", l, r2[tid]);

    for (int s = blockDim.x / 2; s > 0; s = s / 2)
    {
        if (tid < s) r2[tid] += r2[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        //printf("%d %d %f\n", blockIdx.x, threadIdx.x, r2[tid]);
        atomicAdd(&r20, r2[tid]);
    }
}

void JacobiBicgSolverCuda::reset(CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux, CMFD_VAR* src, float& r20) {

    _calpha = 1;
    _crho   = 1;
    _comega = 1;

    r20 = 0.0;
    checkCudaErrors(cudaMemcpy(_r20_dev, &r20, sizeof(float), cudaMemcpyHostToDevice));
    ::reset<<<BLOCKS_NODE, THREADS_NODE>>>(*this, diag, cc, flux, src, *_r20_dev);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(&r20, _r20_dev, sizeof(float), cudaMemcpyDeviceToHost));
    r20 = sqrt(r20);
}

__global__ void minv(JacobiBicgSolverCuda& self, CMFD_VAR* cc, CMFD_VAR* b, SOL_VAR* x) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    Geometry* _g = &self.g();


    x(0, l) = self.delinv(0, 0, l) * b(0, l) + self.delinv(1, 0, l) * b(1, l);
    x(1, l) = self.delinv(0, 1, l) * b(0, l) + self.delinv(1, 1, l) * b(1, l);
}


void JacobiBicgSolverCuda::minv(CMFD_VAR* cc, CMFD_VAR* b, SOL_VAR* x) {

    ::minv << <BLOCKS_NODE, THREADS_NODE >> > (*this, cc, b, x);
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void facilu(JacobiBicgSolverCuda& self, CMFD_VAR* diag, CMFD_VAR* cc) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    Geometry* _g = &self.g();

    invmat2g(&diag(0, 0, l), &self.delinv(0, 0, l));

}
void JacobiBicgSolverCuda::facilu(CMFD_VAR* diag, CMFD_VAR* cc) {

    ::facilu << <BLOCKS_NODE, THREADS_NODE >> > (*this, diag, cc);
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void axb(JacobiBicgSolverCuda& self, CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux, CMFD_VAR* aphi) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    Geometry* _g = &self.g();

    for (int ig = 0; ig < _g->ng(); ++ig) {
        aphi(ig, l) = self.JacobiBicgSolver::axb(ig, l, diag, cc, flux);
    }

}

void JacobiBicgSolverCuda::axb(CMFD_VAR* diag, CMFD_VAR* cc, SOL_VAR* flux, CMFD_VAR* aphi) {
    ::axb << <BLOCKS_NODE, THREADS_NODE >> > (*this, diag, cc, flux, aphi);
    checkCudaErrors(cudaDeviceSynchronize());
}

CMFD_VAR temp1[12532];

void JacobiBicgSolverCuda::solve(CMFD_VAR* diag, CMFD_VAR* cc, float& r20, SOL_VAR* flux, float& r2) {
    int n = _g->nxyz() * _g->ng();

    // solves the linear system by preconditioned BiCGSTAB Algorithm
    CMFD_VAR crhod = _crho;
    myblascuda::dot << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _vr0, _vr, _crho_dev);
    checkCudaErrors(cudaDeviceSynchronize());
    //    checkCudaErrors(cudaMemcpy(temp1, _vr, sizeof(CMFD_VAR) * n, cudaMemcpyDeviceToHost));


    cudaMemcpy(&_crho, _crho_dev, sizeof(CMFD_VAR), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaDeviceSynchronize());

    _cbeta = _crho * _calpha / (crhod * _comega);

    //    _vp(:,:,:)=_vr(:,:,:)+_cbeta*(_vp(:,:,:)-_comega*_vv(:,:,:))
    myblascuda::multi << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _comega, _vv, _vt);
    checkCudaErrors(cudaDeviceSynchronize());
    //    checkCudaErrors(cudaMemcpy(temp1, _vt, sizeof(CMFD_VAR) * n, cudaMemcpyDeviceToHost));

    myblascuda::minus << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _vp, _vt, _vt);
    checkCudaErrors(cudaDeviceSynchronize());

    myblascuda::multi << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _cbeta, _vt, _vt);
    checkCudaErrors(cudaDeviceSynchronize());

    myblascuda::plus << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _vr, _vt, _vp);
    checkCudaErrors(cudaDeviceSynchronize());
    //    checkCudaErrors(cudaMemcpy(temp1, _vr, sizeof(CMFD_VAR) * n, cudaMemcpyDeviceToHost));
    //    checkCudaErrors(cudaMemcpy(temp1, _vt, sizeof(CMFD_VAR) * n, cudaMemcpyDeviceToHost));
    //    checkCudaErrors(cudaMemcpy(temp1, _vp, sizeof(CMFD_VAR) * n, cudaMemcpyDeviceToHost));

    minv(cc, _vp, _vy);
    //    checkCudaErrors(cudaMemcpy(temp1, _vy, sizeof(CMFD_VAR) * n, cudaMemcpyDeviceToHost));

    axb(diag, cc, _vy, _vv);
    //    checkCudaErrors(cudaMemcpy(temp1, _vv, sizeof(CMFD_VAR) * n, cudaMemcpyDeviceToHost));

    CMFD_VAR r0v;
    myblascuda::dot << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _vr0, _vv, _r0v_dev);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(&r0v, _r0v_dev, sizeof(CMFD_VAR), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaDeviceSynchronize());

    if (r0v == 0.0) {
        return;
    }

    _calpha = _crho / r0v;

    //    _vs(:,:,:)=_vr(:,:,:)-_calpha*_vv(:,:,:)
    myblascuda::multi << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _calpha, _vv, _vt);
    checkCudaErrors(cudaDeviceSynchronize());
    //    checkCudaErrors(cudaMemcpy(temp1, _vt, sizeof(CMFD_VAR) * n, cudaMemcpyDeviceToHost));

    myblascuda::minus << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _vr, _vt, _vs);
    checkCudaErrors(cudaDeviceSynchronize());
    //    checkCudaErrors(cudaMemcpy(temp1, _vs, sizeof(CMFD_VAR) * n, cudaMemcpyDeviceToHost));

    minv(cc, _vs, _vz);
    axb(diag, cc, _vz, _vt);

    CMFD_VAR pts, ptt;
    myblascuda::dot << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _vs, _vt, _pts_dev);
    checkCudaErrors(cudaDeviceSynchronize());

    myblascuda::dot << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _vt, _vt, _ptt_dev);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(&pts, _pts_dev, sizeof(CMFD_VAR), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ptt, _ptt_dev, sizeof(CMFD_VAR), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaDeviceSynchronize());


    _comega = 0.0;
    if (ptt != 0.0) {
        _comega = pts / ptt;
    }

    //    flux(:, :, :) = flux(:, :, :) + _calpha * _vy(:,:,:)+_comega * _vz(:,:,:)
    myblascuda::multi << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _comega, _vz, _vz);
    checkCudaErrors(cudaDeviceSynchronize());

    myblascuda::multi << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _calpha, _vy, _vy);
    checkCudaErrors(cudaDeviceSynchronize());

    myblascuda::plus << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _vz, _vy, _vy);
    checkCudaErrors(cudaDeviceSynchronize());

    myblascuda::plus << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, flux, _vy, flux);
    checkCudaErrors(cudaDeviceSynchronize());


    //    _vr(:,:,:)=_vs(:,:,:)-_comega * _vt(:,:,:)
    myblascuda::multi << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _comega, _vt, _vr);
    checkCudaErrors(cudaDeviceSynchronize());

    myblascuda::minus << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _vs, _vr, _vr);
    checkCudaErrors(cudaDeviceSynchronize());

    if (r20 != 0.0) {
        r2 = 0.0;
        checkCudaErrors(cudaMemcpy(_r2_dev, &r2, sizeof(float), cudaMemcpyHostToDevice));
        myblascuda::dot << <BLOCKS_NGXYZ, THREADS_NGXYZ >> > (n, _vt, _vt, _r2_dev);
        checkCudaErrors(cudaDeviceSynchronize());

        cudaMemcpy(&r2, _r2_dev, sizeof(CMFD_VAR), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaDeviceSynchronize());

        r2 = sqrt(r2) / r20;
    }
}