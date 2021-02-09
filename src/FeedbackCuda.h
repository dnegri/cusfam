#pragma once

#include "pch.h"
#include "GeometryCuda.h"
#include "Feedback.h"
#include "SteamTableCuda.h"

class FeedbackCuda : public Feedback {

public:
    __host__ __device__ FeedbackCuda(GeometryCuda& g, SteamTableCuda& steam, Feedback& f);
    __host__ __device__ virtual ~FeedbackCuda();

    __host__ __device__ void allocate(const int& nft);

};


