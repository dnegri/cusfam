#pragma once

#include "pch.h"
#include "GeometryCuda.h"
#include "Feedback.h"
#include "SteamTableCuda.h"

class FeedbackCuda : public Feedback {
private:
    SteamTable* _steam_cpu;
public:
    __host__ __device__ FeedbackCuda(GeometryCuda& g, SteamTableCuda& steam);
    __host__ __device__ virtual ~FeedbackCuda();

    __host__ __device__ void allocate();
    __host__ __device__ void updateTin(const float& tin);
    __host__ void copyFeedback(Feedback& f);
    __host__ void setTf(const float* tf);
    __host__ void setTm(const float* tm);
    __host__ void setDm(const float* dm);


};


