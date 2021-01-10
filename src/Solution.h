#pragma once
#include "pch.h"

#define SOL_PRECISION  float

class Solution {
private:
    int _ng;
    int _nxyz;
    int _nsurf;
    SOL_PRECISION* _phi;
    SOL_PRECISION* _psi;
    SOL_PRECISION* _jnet;

public:
    Solution(int ng, int nxyz) {
        _ng = ng;
        _nxyz = nxyz;

        _phi = new SOL_PRECISION[_ng*_nxyz]();
        _psi = new SOL_PRECISION[_ng*_nxyz]();
        _jnet = new SOL_PRECISION[_ng*_nxyz]();
    };

    virtual ~Solution(){
        delete[] _phi;
        delete[] _psi;
        delete[] _jnet;
    };

    inline SOL_PRECISION& phi(const int & ig, const int & l)    {return _phi[l*_ng+ig];};
    inline SOL_PRECISION& psi(const int & ig, const int & l)    {return _psi[l];};
    inline SOL_PRECISION& jnet(const int & ig, const int & ls)  {return _jnet[ls*_ng+ig];};

};

