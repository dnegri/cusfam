#pragma once
#include "pch.h"
#include "Geometry.h"
#include "CrossSection.h"
#include "CSRSolver.h"


class CMFD : public Managed {
protected:
    Geometry& _g;
    CrossSection& _x;
    CSRSolver* _ls;

	int _ng;
	int _ncmfd;

	float* _dtil;
	float* _dhat;
	float* _diag;
	float* _cc;
	double* _src;

public:
    void setNcmfd(int ncmfd);

    void setEshift(float eshift0);

    void setEpsl2(float epsl2);

protected:
    float _eshift;
    float _epsl2;



public:
    __host__ CMFD(Geometry& g, CrossSection& x);
    __host__ virtual ~CMFD();

    __host__ __device__ virtual void upddtil()=0;
    __host__ __device__ virtual void upddhat(double* flux, float* jnet)=0;
    __host__ __device__ virtual void setls()=0;

	__host__ __device__ void upddtil(const int& ls);
    __host__ __device__ void upddhat(const int& ls, double* flux, float* jnet);
    __host__ __device__ void setls(const int& l);

    __host__ __device__ float& dtil(const int& ig, const int& ls) {return _dtil[ls*_g.ng()+ig];};
    __host__ __device__ float& dhat(const int& ig, const int& ls) {return _dhat[ls*_g.ng()+ig];};
    __host__ __device__ float& diag(const int& igs, const int& ige, const int& l) {return _diag[l*_g.ng2()+ige*_g.ng()+igs];};
    __host__ __device__ float& cc(const int& lr,const int& idir, const int& ig, const int& l) {
        return _cc[l*_g.ng()*NDIRMAX*LR+ig*NDIRMAX*LR+idir*LR+lr];
    };
    __host__ __device__ double& src(const int& ig, const int& l) {return _src[l*_g.ng()+ig];};

    __host__ __device__ double axb(const int& ig, const int& l, const double* flux) {

        double ab = 0.0;
        for (int igs = 0; igs < _g.ng(); ++igs) {
            ab += diag(igs, ig, l)*flux[l*_g.ng()+igs];
        }

        for (int idir = 0; idir < NDIRMAX; ++idir) {
            for (int lr = 0; lr < LR; ++lr) {
                int ln = _g.neib(lr, idir, l);
                if (ln != -1)
                    ab += cc(lr, idir, ig, l) * flux[ln*_g.ng()+ ig];
            }
        }

        return ab;
    };
};