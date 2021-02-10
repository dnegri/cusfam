#include "SimonCPU.h"
#include "myblas.h"


SimonCPU::SimonCPU() {
}

SimonCPU::~SimonCPU() {

}

void SimonCPU::initialize(const char* dbfile)
{
    Simon::initialize(dbfile);
    _cmfd = new BICGCMFD(g(), x());
    cmfd().setNcmfd(3);
    cmfd().setEshift(0.04);
}

void SimonCPU::runKeff(const int& nmaxout) {
    float errl2 = 0.0;
    int nboiling = 0;
    cmfd().setNcmfd(nmaxout);

    cmfd().updpsi(_flux);

    _ppm = 100.0;
    f().updatePPM(_ppm);
    d().updateH2ODensity(f().dm(), _ppm);
    x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());
    cmfd().upddtil();
    cmfd().setls(_eigv);
    cmfd().drive(_eigv, _flux, errl2);

    _ppm = 1000.0;
    f().updatePPM(_ppm);
    d().updateH2ODensity(f().dm(), _ppm);
    x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());
    cmfd().upddtil();
    cmfd().setls(_eigv);
    cmfd().drive(_eigv, _flux, errl2);
    exit(0);
    normalize();
}

void SimonCPU::runECP(const int& nmaxout, const double& eigvt) {
    float errl2 = 0.0;
    int nboiling = 0;

    cmfd().setNcmfd(3);
    cmfd().updpsi(_flux);

    float ppmd = _ppm;
    double eigvd = _eigv;

    for (size_t iout = 0; iout < nmaxout; iout++)
    {
        f().updatePPM(_ppm);
        d().updateH2ODensity(f().dm(), _ppm);
        x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());
        cmfd().upddtil();
        cmfd().setls(_eigv);
        cmfd().drive(_eigv, _flux, errl2);
        normalize();

        if (iout > 3 && errl2 < 1.E-5) break;


        double temp = _ppm;

        if (iout == 0)
            _ppm = _ppm + (_eigv - eigvt) * 1E5 / 10.0;
        else
            _ppm = (_ppm - ppmd) / (_eigv - eigvd) * (eigvt - _eigv) + _ppm;

        if(_ppm > temp+300.0) {
            _ppm = temp+300.0;
        } else if(_ppm < temp-300.0) {
            _ppm = temp-300.0;
        }

        ppmd = temp;
        eigvd = _eigv;

        printf("CHANGE PPM : %.2f --> %.2f\n", ppmd, _ppm);

        //search critical
        f().updateTm(_power, nboiling);
        f().updateTf(_power, d().burn());
        d().eqxe(x().xsmica(), x().xsmicf(), _flux, _fnorm);
    }
}

void SimonCPU::runDepletion(const float& dburn) {

    d().pickData(x().xsmica(), x().xsmicf(), x().xsmic2n(), _flux, _fnorm);
    d().dep(116748.0);
}

void SimonCPU::runXenonTransient() {

}

void SimonCPU::normalize()
{
    double ptotal = 0;
    for (size_t l = 0; l < _g->nxyz(); l++)
    {
        power(l) = 0.0;
        for (size_t ig = 0; ig < _g->ng(); ig++)
        {
            power(l) += flux(ig, l) * x().xskf(ig, l);
        }
        power(l) *= _g->vol(l);
        ptotal += power(l);
    }

    _fnorm = _pload * 0.25 / ptotal;

    for (size_t l = 0; l < _g->nxyz(); l++)
    {
        power(l) *= _fnorm;
    }

}
