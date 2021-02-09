//
// Created by JOO IL YOON on 2021/01/25.
//

#include "FeedbackCuda.h"

#define pow2(l2d, k)  pow[k*_g.nxy()+l2d]
#define pow3(l)  pow[l]
#define bu(l)  bu[l]


FeedbackCuda::FeedbackCuda(GeometryCuda& g, SteamTableCuda& steam, Feedback& f) : Feedback(g, steam) {

    _tffrz = f.tffrz();
    _tmfrz = f.tmfrz();
    _dmfrz = f.dmfrz();
    _heatfrac = f.heatfrac();
    _hin = f.hin();
    _din = f.din();
    _tin = f.tin();
    _nft = f.nft();

    checkCudaErrors(cudaMalloc((void**)&_tf, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_tm, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_dm, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_dppm, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_dtf, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_dtm, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_ddm, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_ppm0, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_stf0, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_tm0, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_dm0, sizeof(float) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_chflow, sizeof(float) * g.nxy()));
    checkCudaErrors(cudaMalloc((void**)&_fueltype, sizeof(int) * g.nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_frodn, sizeof(float) * g.nxy()));
    checkCudaErrors(cudaMemcpy(_tf, f.tf(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_tm, f.tm(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dm, f.dm(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dppm, f.dppm(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dtf, f.dtf(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dtm, f.dtm(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_ddm, f.ddm(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_ppm0, f.ppm0(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_stf0, f.stf0(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_tm0, f.tm0(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dm0, f.dm0(), sizeof(float) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_chflow, f.chflow(), sizeof(float) * g.nxy(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_fueltype, f.fueltype(), sizeof(int) * g.nxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_frodn, f.frodn(), sizeof(float) * g.nxy(), cudaMemcpyHostToDevice));
}

FeedbackCuda::~FeedbackCuda()
{
}