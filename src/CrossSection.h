#pragma once
#include "pch.h"

#ifndef XS_PRECISION
    #define XS_PRECISION double
#endif

class CrossSection : public Managed {
private:
    int _ng;
    int _nxyz;
    XS_PRECISION* _xsnf;
    XS_PRECISION* _xsdf;
    XS_PRECISION* _xssf;
    XS_PRECISION* _xstf;
    XS_PRECISION* _xskf;
    XS_PRECISION* _chif;
    XS_PRECISION* _xsadf;
public:
    __host__ CrossSection(const int& ng, const int& nxyz, XS_PRECISION* xsdf, XS_PRECISION* xstf, XS_PRECISION* xsnf, XS_PRECISION* xssf, XS_PRECISION* xschif, XS_PRECISION* xsadf) {
        _ng = ng;
        _nxyz = nxyz;
        _xsnf = xsnf;
        _xsdf = xsdf;
        _xstf = xstf;
        _chif = xschif;
        _xssf = xssf;
        _xsadf = xsadf;
    };

    __host__ CrossSection(const int& ng, const int& nxyz) {
        _ng = ng;
        _nxyz = nxyz;
        _xsnf = new XS_PRECISION[_ng * _nxyz]{};
        _xsdf = new XS_PRECISION[_ng*_nxyz]{};
        _xstf = new XS_PRECISION[_ng*_nxyz]{};
        _xskf = new XS_PRECISION[_ng*_nxyz]{};
        _chif = new XS_PRECISION[_ng*_nxyz]{};
        _xssf = new XS_PRECISION[_ng*_ng*_nxyz]{};
        _xsadf = new XS_PRECISION[_ng * _nxyz]{};
    };

    __host__ virtual ~CrossSection() {
        delete [] _xsnf;
        delete [] _xsdf;
        delete [] _xstf;
        delete [] _xskf;
        delete [] _chif;
        delete [] _xssf;
    }

    __host__ __device__ inline XS_PRECISION& xsnf(const int & ig, const int & l) {return _xsnf[l*_ng+ig];};
    __host__ __device__ inline XS_PRECISION& xsdf(const int & ig, const int & l) {return _xsdf[l*_ng+ig];};
    __host__ __device__ inline XS_PRECISION& xssf(const int & igs, const int & ige, const int & l) {return _xssf[l*_ng*_ng+ige*_ng+igs];};
    __host__ __device__ inline XS_PRECISION& xstf(const int & ig, const int & l) {return _xstf[l*_ng+ig];};
    __host__ __device__ inline XS_PRECISION& xskf(const int & ig, const int & l) {return _xskf[l*_ng+ig];};
    __host__ __device__ inline XS_PRECISION& chif(const int & ig, const int & l) {return _chif[l*_ng+ig];};
    __host__ __device__ inline XS_PRECISION& xsadf(const int& ig, const int& l) { return _xsadf[l * _ng + ig]; };

};
