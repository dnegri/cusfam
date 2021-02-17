//
// Created by JOO IL YOON on 2021/01/25.
//

#include "FeedbackCuda.h"

#define pow2(l2d, k)  pow[k*_g.nxy()+l2d]
#define pow3(l)  pow[l]
#define bu(l)  bu[l]


FeedbackCuda::FeedbackCuda(GeometryCuda& g, SteamTableCuda& steam) : Feedback(g, steam) {
    _steam_cpu = &steam.getSteamTableCPU();
}

FeedbackCuda::FeedbackCuda(const FeedbackCuda& f) : Feedback(f)
{
    printf("copy contructor of FeedbackCuda is called.");
}

void FeedbackCuda::allocate() {

    checkCudaErrors(cudaMalloc((void**)&_tf, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_tm, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_dm, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_dppm, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_dtf, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_dtm, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_ddm, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_ppm0, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_stf0, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_tm0, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_dm0, sizeof(float) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_chflow, sizeof(float) * _g.nxy()));
    checkCudaErrors(cudaMalloc((void**)&_fueltype, sizeof(int) * _g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_frodn, sizeof(float) * _g.nxy()));

    checkCudaErrors(cudaDeviceSynchronize());
}

FeedbackCuda::~FeedbackCuda()
{
}

void FeedbackCuda::copyFeedback(Feedback& f)
{
    _tffrz = f.tffrz();
    _tmfrz = f.tmfrz();
    _dmfrz = f.dmfrz();
    _heatfrac = f.heatfrac();
    _hin = f.hin();
    _din = f.din();
    _tin = f.tin();
    _nft = f.nft();

    checkCudaErrors(cudaMemcpy(_ppm0, f.ppm0(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_stf0, f.stf0(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_tm0, f.tm0(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dm0, f.dm0(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_chflow, f.chflow(), sizeof(float) * _g.nxy(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_fueltype, f.fueltype(), sizeof(int) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_frodn, f.frodn(), sizeof(float) * _g.nxy(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_tf, f.tf(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_tm, f.tm(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dm, f.dm(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dppm, f.dppm(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dtf, f.dtf(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dtm, f.dtm(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_ddm, f.ddm(), sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
}

void FeedbackCuda::updateTin(const float& tin)
{
    _tin = tin;
    _steam_cpu->getEnthalpy(tin, _hin);
    _steam_cpu->getDensity(tin, _din);
}

void FeedbackCuda::setTf(const float* tf)
{
    checkCudaErrors(cudaMemcpy(_tf, tf, sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
}

void FeedbackCuda::setTm(const float* tm)
{
    checkCudaErrors(cudaMemcpy(_tm, tm, sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
}

void FeedbackCuda::setDm(const float* dm)
{
    checkCudaErrors(cudaMemcpy(_dm, dm, sizeof(float) * _g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void updateTf(FeedbackCuda& self, const float* power, const float* burnup, float heatfrac)
{
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    self.Feedback::updateTf(l, power, burnup, heatfrac);
}

void FeedbackCuda::updateTf(const float* power, const float* burnup)
{
    ::updateTf<<<BLOCKS_NODE, THREADS_NODE>>>(*this, power, burnup, _heatfrac);
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void updateTm(FeedbackCuda& self, const float* power, float hin, float tin, float din)
{
    int l2d = threadIdx.x + blockIdx.x * blockDim.x;
    if (l2d >= self.g().nxy()) return;

    int nboiling = 0;
    self.Feedback::updateTm(l2d, power, hin, tin, din,nboiling);
}


void FeedbackCuda::updateTm(const float* power, int& nboiling)
{
    ::updateTm<<<BLOCKS_2D, THREADS_2D>>>(*this, power, _hin, _tin, _din);
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void updatePPM(FeedbackCuda& self, float ppm) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    self.dppm(l) = ppm * self.dm(l)/self.dm0(l) - self.ppm0(l);
}


void FeedbackCuda::updatePPM(const float& ppm) {
    ::updatePPM<<<BLOCKS_NODE, THREADS_NODE>>>(*this, ppm);
    checkCudaErrors(cudaDeviceSynchronize());

}

__global__ void initDelta(FeedbackCuda& self, float ppm) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l >= self.g().nxyz()) return;

    self.dppm(l) = ppm * self.dm(l)/self.dm0(l) - self.ppm0(l);
    self.dtf(l) = sqrt(self.tf(l)) - self.stf0(l);
    self.dtm(l) = self.tm(l) - self.tm0(l);
    self.ddm(l) = self.dm(l) - self.dm0(l);

}


void FeedbackCuda::initDelta(const float& ppm) {
    ::initDelta<<<BLOCKS_NODE, THREADS_NODE>>>(*this, ppm);
    checkCudaErrors(cudaDeviceSynchronize());

}

