#pragma once
#include "pch.h"
#include "Geometry.h"
#include "CrossSection.h"


class CMFD : public Managed {
protected:
    Geometry& _g;
    CrossSection& _x;

	int _ng;
	int _ncmfd;

	float* _dtil;
	float* _dhat;
	float* _am;
	float* _af;
	float* _cc;
	float* _src;
    float* _psi;

    float _eshift0  =   0.04;
    float _eshift   =   0.04;
    float _eigvs;
    float _reigvs;
    float _reigvsd;
    float _eigshft;
    int   _nin2g = 0;

    float _epsl2;



public:
    __host__ CMFD(Geometry& g, CrossSection& x);
    __host__ virtual ~CMFD();

    __host__ __device__ virtual void upddtil()=0;
    __host__ __device__ virtual void upddhat()=0;
    __host__ __device__ virtual void setls()=0;

	__host__ __device__ void upddtil(const int& ls);
    __host__ __device__ void upddhat(const int& ls, float* flux, float* jnet);
    __host__ __device__ void setls(const int& l);

    __host__ __device__ float& dtil(const int& ig, const int& ls) {return _dtil[ls*_g.ng()+ig];};
    __host__ __device__ float& dhat(const int& ig, const int& ls) {return _dhat[ls*_g.ng()+ig];};
    __host__ __device__ float& am(const int& igs, const int& ige, const int& l) {return _dhat[l*_g.ng2()+ige*_g.ng()+igs];};
    __host__ __device__ float& cc(const int& lr,const int& idir, const int& ig, const int& l) {
        return _cc[l*_g.ng()*NDIRMAX*LR+ig*NDIRMAX*LR+idir*LR+lr];
    };
    __host__ __device__ float& af(const int& ig, const int& l) {return _af[l*_g.ng()+ig];};
    __host__ __device__ float& psi(const int& l) {return _psi[l];};
    __host__ __device__ float& src(const int& ig, const int& l) {return _src[l*_g.ng()+ig];};

};