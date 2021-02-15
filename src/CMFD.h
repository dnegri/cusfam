#pragma once
#include "pch.h"
#include "Geometry.h"
#include "CrossSection.h"
#include "CSRSolver.h"


class CMFD : public Managed {
protected:
    Geometry& _g;
    CrossSection& _x;

	int _ncmfd;

    CMFD_VAR* _dtil;
    CMFD_VAR* _dhat;
    CMFD_VAR* _diag;
    CMFD_VAR* _cc;
    CMFD_VAR* _src;
    CMFD_VAR* _psi;

    float _epsl2;


public:
    __host__ CMFD(Geometry& g, CrossSection& x);
    __host__ virtual ~CMFD();

    __host__ __device__ Geometry& g() { return _g; };
    __host__ __device__ CrossSection& x() { return _x; };

    __host__ virtual void upddtil()=0;
    __host__ virtual void upddhat(SOL_VAR* flux, SOL_VAR* jnet)=0;
    __host__ virtual void setls(const double& eigv)=0;
    __host__ virtual void updjnet(SOL_VAR* flux, SOL_VAR* jnet)=0;
    __host__ virtual void updpsi(const SOL_VAR* flux)=0;
    __host__ virtual void drive(double& eigv, SOL_VAR* flux, float& errl2)=0;

    __host__ virtual void init();

    __host__ __device__ void updpsi(const int& l, const SOL_VAR* flux);
	__host__ __device__ void upddtil(const int& ls);
    __host__ __device__ void upddhat(const int& ls, SOL_VAR* flux, SOL_VAR* jnet);
    __host__ __device__ void setls(const int& l);

    __host__ __device__ void setNcmfd(int ncmfd);
    __host__ __device__ void setEpsl2(float epsl2);
    __host__ __device__ CMFD_VAR& dtil(const int& ig, const int& ls) {return _dtil[ls*_g.ng()+ig];};
    __host__ __device__ CMFD_VAR& dhat(const int& ig, const int& ls) {return _dhat[ls*_g.ng()+ig];};
    __host__ __device__ CMFD_VAR& diag(const int& igs, const int& ige, const int& l) {return _diag[l*_g.ng2()+ige*_g.ng()+igs];};
    __host__ __device__ CMFD_VAR& cc(const int& lr,const int& idir, const int& ig, const int& l) {
        return _cc[l*_g.ng()*NDIRMAX*LR+ig*NDIRMAX*LR+idir*LR+lr];
    };
    __host__ __device__ CMFD_VAR& src(const int& ig, const int& l) {return _src[l*_g.ng()+ig];};
    __host__ __device__ CMFD_VAR& psi(const int& l) {return _psi[l];};

    __host__ __device__ CMFD_VAR axb(const int& ig, const int& l, const SOL_VAR* flux) {

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

    __host__ __device__ void updjnet(const int& ls, const SOL_VAR* flux, SOL_VAR* jnet) {
        int ll = _g.lklr(LEFT, ls);
        int lr = _g.lklr(RIGHT, ls);
        int idirl = _g.idirlr(LEFT, ls);
        int idirr = _g.idirlr(RIGHT, ls);

        for (int ig = 0; ig < _g.ng(); ig++)
        {
            if (ll < 0) {
                jnet[ls*_g.ng()+ig] = -(dtil(ig, ls) + dhat(ig, ls)) * flux[lr*_g.ng()+ig];
            }
            else if (lr < 0) {
                jnet[ls * _g.ng() + ig] = (dtil(ig, ls) - dhat(ig, ls)) * flux[ll * _g.ng() + ig];
            }
            else {
                jnet[ls * _g.ng() + ig] = -dtil(ig, ls) * (flux[lr * _g.ng() + ig] - flux[ll * _g.ng() + ig])
                                          -dhat(ig, ls) * (flux[lr * _g.ng() + ig] + flux[ll * _g.ng() + ig]);
            }
        }

    }


};