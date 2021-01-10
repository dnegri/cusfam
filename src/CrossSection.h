#pragma once
#include "pch.h"

#ifndef XS_PRECISION
    #define XS_PRECISION float
#endif

class CrossSection {
private:
    int _ng;
    int _nxyz;
    XS_PRECISION* _xsnf;
    XS_PRECISION* _xsdf;
    XS_PRECISION* _xssf;
    XS_PRECISION* _xstf;
    XS_PRECISION* _xskf;
    XS_PRECISION* _chif;

public:
    CrossSection(int ng, int nxyz) {
        _xsnf = new XS_PRECISION[_ng*_nxyz]();
        _xsdf = new XS_PRECISION[_ng*_nxyz]();
        _xstf = new XS_PRECISION[_ng*_nxyz]();
        _xskf = new XS_PRECISION[_ng*_nxyz]();
        _chif = new XS_PRECISION[_ng*_nxyz]();
        _xssf = new XS_PRECISION[_ng*_ng*_nxyz]();
    };
    
    virtual ~CrossSection() {
        delete [] _xsnf;
        delete [] _xsdf;
        delete [] _xstf;
        delete [] _xskf;
        delete [] _chif;
        delete [] _xssf;
    }

    inline XS_PRECISION& xsnf(const int & ig, const int & l) {return _xsnf[l*_ng+ig];};
    inline XS_PRECISION& xsdf(const int & ig, const int & l) {return _xsdf[l*_ng+ig];};
    inline XS_PRECISION& xssf(const int & igs, const int & ige, const int & l) {return _xsnf[l*_ng*_ng+igs*_ng+ige];};
    inline XS_PRECISION& xstf(const int & ig, const int & l) {return _xstf[l*_ng+ig];};
    inline XS_PRECISION& xskf(const int & ig, const int & l) {return _xskf[l*_ng+ig];};
    inline XS_PRECISION& chif(const int & ig, const int & l) {return _chif[l*_ng+ig];};

};
