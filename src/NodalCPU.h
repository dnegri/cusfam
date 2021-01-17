#pragma once
#include "Nodal.h"
#include "CrossSection.h"

class NodalCPU : public Nodal {
private:

    CrossSection& xs;

    int _ng;
    int _ng2;
    int _nxyz;
    int _nsurf;

    int _symopt;
    int _symang;
    float* _albedo;

    int* _neib;
    int* _lktosfc;
    float* _hmesh;

    int* _lklr;
    int* _idirlr;
    int* _sgnlr;

    XS_PRECISION* _xstf;
    XS_PRECISION* _xsdf;
    XS_PRECISION* _xsnff;
    XS_PRECISION* _xschif;
    XS_PRECISION* _xssf;
    XS_PRECISION* _xsadf;

    float* _trlcff0;
    float* _trlcff1;
    float* _trlcff2;
    float* _eta1;
    float* _eta2;
    float* _mu;
    float* _tau;


    float* _m260;
    float* _m251;
    float* _m253;
    float* _m262;
    float* _m264;

    float* _diagDI;
    float* _diagD;
    float* _matM;
    float* _matMI;
    float* _matMs;
    float* _matMf;

    float* _dsncff2;
    float* _dsncff4;
    float* _dsncff6;

    float* _jnet;
    double* _flux;
    double _reigv;

public:
    NodalCPU(Geometry& g, CrossSection& xs);

    virtual ~NodalCPU();

    inline float& jnet(const int& ig, const int& lks) { return _jnet[lks * _ng + ig]; };
    inline double& flux(const int& ig, const int& lk) { return _flux[lk * _ng + ig]; };

    void init();
    void reset(CrossSection& xs, double* reigv, double* jnet, double* phif);
    void drive();

};


