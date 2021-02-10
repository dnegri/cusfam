//
// Created by JOO IL YOON on 2021/01/30.
//

#include "JacobiBicgSolverCuda.h"
#include "mat2g.h"
#include "myblas.h"
#include <cuda.h>
#include <device_functions.h>
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
#define phi(ig, l) phi[l*_g->ng()+ig]
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

    checkCudaErrors(cudaMalloc((void**)&_vz, sizeof(double) * _g->ng() * _g->nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vy, sizeof(double) * _g->ng() * _g->nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vr, sizeof(CMFD_VAR) * _g->ng() * _g->nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vr0, sizeof(CMFD_VAR) * _g->ng() * _g->nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vp, sizeof(CMFD_VAR) * _g->ng() * _g->nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vv, sizeof(CMFD_VAR) * _g->ng() * _g->nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vs, sizeof(CMFD_VAR) * _g->ng() * _g->nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_vt, sizeof(CMFD_VAR) * _g->ng() * _g->nxyz()));
}

JacobiBicgSolverCuda::~JacobiBicgSolverCuda() {

}

__global__ void reset(JacobiBicgSolverCuda& self, CMFD_VAR* diag, CMFD_VAR* cc, double* phi, CMFD_VAR* src, float& r20) {

    self.alpha() = 1;
    self.rho() = 1;
    self.omega() = 1;

    r20 = 0;
    __shared__ int r2[NTHREADSPERBLOCK];

    int tid = threadIdx.x;
    int l = tid + blockIdx.x * blockDim.x;

    r2[tid] = self.JacobiBicgSolver::reset(l, diag, cc, phi, src);

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s = s / 2)
    {
        if (tid < s) r2[tid] += r2[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(&r20, r2[tid]);
}

void JacobiBicgSolverCuda::reset(CMFD_VAR* diag, CMFD_VAR* cc, double* phi, CMFD_VAR* src, float& r20) {

    ::reset<<<BLOCKS_NODE, THREADS_NODE>>>(*this, diag, cc, phi, src, r20);
}

__global__ void minv(JacobiBicgSolverCuda& self, CMFD_VAR* cc, CMFD_VAR* b, double* x) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    Geometry* _g = &self.g();


    x(0, l) = self.delinv(0, 0, l) * b(0, l) + self.delinv(1, 0, l) * b(1, l);
    x(1, l) = self.delinv(0, 1, l) * b(0, l) + self.delinv(1, 1, l) * b(1, l);
}


void JacobiBicgSolverCuda::minv(CMFD_VAR* cc, CMFD_VAR* b, double* x) {

    ::minv << <BLOCKS_NODE, THREADS_NODE >> > (*this, cc, b, x);
}

__global__ void facilu(JacobiBicgSolverCuda& self, CMFD_VAR* diag, CMFD_VAR* cc) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    Geometry* _g = &self.g();

    invmat2g(&diag(0, 0, l), &self.delinv(0, 0, l));

}
void JacobiBicgSolverCuda::facilu(CMFD_VAR* diag, CMFD_VAR* cc) {

    ::facilu << <BLOCKS_NODE, THREADS_NODE >> > (*this, diag, cc);
}

__global__ void axb(JacobiBicgSolverCuda& self, CMFD_VAR* diag, CMFD_VAR* cc, double* phi, CMFD_VAR* aphi) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    Geometry* _g = &self.g();

    for (int ig = 0; ig < _g->ng(); ++ig) {
        aphi(ig, l) = self.JacobiBicgSolver::axb(ig, l, diag, cc, phi);
    }

}

void JacobiBicgSolverCuda::axb(CMFD_VAR* diag, CMFD_VAR* cc, double* phi, CMFD_VAR* aphi) {
    ::axb << <BLOCKS_NODE, THREADS_NODE >> > (*this, diag, cc, phi, aphi);
}


void JacobiBicgSolverCuda::solve(CMFD_VAR* diag, CMFD_VAR* cc, CMFD_VAR& r20, double* phi, double& r2) {
    int n = _g->nxyz() * _g->ng();

    // solves the linear system by preconditioned BiCGSTAB Algorithm
    double crhod = _crho;
    _crho = myblas::dot(n, _vr0, _vr);
    _cbeta = _crho * _calpha / (crhod * _comega);

    //    _vp(:,:,:)=_vr(:,:,:)+_cbeta*(_vp(:,:,:)-_comega*_vv(:,:,:))
    myblas::multi(n, _comega, _vv, _vt);
    myblas::minus(n, _vp, _vt, _vt);
    myblas::multi(n, _cbeta, _vt, _vt);
    myblas::plus(n, _vr, _vt, _vp);

    minv(cc, _vp, _vy);
    axb(diag, cc, _vy, _vv);

    CMFD_VAR r0v = myblas::dot(n, _vr0, _vv);

    if (r0v == 0.0) {
        return;
    }

    _calpha = _crho / r0v;

    //    _vs(:,:,:)=_vr(:,:,:)-_calpha*_vv(:,:,:)
    myblas::multi(n, _calpha, _vv, _vt);
    myblas::minus(n, _vr, _vt, _vs);

    minv(cc, _vs, _vz);
    axb(diag, cc, _vz, _vt);

    CMFD_VAR pts = myblas::dot(n, _vs, _vt);
    CMFD_VAR ptt = myblas::dot(n, _vt, _vt);

    _comega = 0.0;
    if (ptt != 0.0) {
        _comega = pts / ptt;
    }

    //    phi(:, :, :) = phi(:, :, :) + _calpha * _vy(:,:,:)+_comega * _vz(:,:,:)
    myblas::multi(n, _comega, _vz, _vz);
    myblas::multi(n, _calpha, _vy, _vy);
    myblas::plus(n, _vz, _vy, _vy);
    myblas::plus(n, phi, _vy, phi);


    //    _vr(:,:,:)=_vs(:,:,:)-_comega * _vt(:,:,:)
    myblas::multi(n, _comega, _vt, _vr);
    myblas::minus(n, _vs, _vr, _vr);

    if (r20 != 0.0) {
        r2 = sqrt(myblas::dot(n, _vt, _vt)) / r20;
    }
}