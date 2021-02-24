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
    _cmfd->init();
    cmfd().setNcmfd(4);
    cmfd().setEshift(0.1);

}

void SimonCPU::runKeff(const int& nmaxout) {
    float errl2 = 0.0;
    int nboiling = 0;

    cmfd().updpsi(_flux);

    for (int i = 0; i < nmaxout; ++i) {
		f().updatePPM(_ppm);
		d().updateH2ODensity(f().dm(), _ppm);
		x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());

		cmfd().upddtil();
		cmfd().setls(_eigv);
        cmfd().drive(_eigv, _flux, errl2);
		normalize();

		f().updateTm(_power, nboiling);
		f().updateTf(_power, d().burn());

		if (errl2 < 1.E-4) {
			printf("XENON UPDATE\n");
			d().eqxe(x().xsmica(), x().xsmicf(), _flux, _fnorm);
		}

		if (i > 3 && errl2 < 1.E-5) break;

		double reigv = 1. / _eigv;
		cmfd().updnodal(reigv, _flux, _jnet);
	}
}

void SimonCPU::runECP(const int& nmaxout, const double& eigvt) {
    float errl2 = 0.0;
    int nboiling = 0;

    cmfd().setNcmfd(5);
    cmfd().updpsi(_flux);

    float ppmd = _ppm;
    double eigvd = _eigv;
    double reigv = 1./_eigv;
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

		if (errl2 < 1.E-4) {
			printf("XENON UPDATE\n");
			d().eqxe(x().xsmica(), x().xsmicf(), _flux, _fnorm);
			reigv = 1./_eigv;
			cmfd().updnodal(reigv, _flux, _jnet);
		}

		//search critical
		f().updateTm(_power, nboiling);
		f().updateTf(_power, d().burn());
		
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

        //nodal
    }

//    for (int l = 0; l < _g->nxy(); ++l) {
//        for (int ig = 0; ig < _g->ng(); ++ig) {
//            printf("FLUX : %e\n", flux(ig,l)*_fnorm);
//        }
//    }
}

void SimonCPU::runDepletion(const float& tsec) {

    d().pickData(x().xsmica(), x().xsmicf(), x().xsmic2n(), _flux, _fnorm);
    d().dep(tsec);
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

    _fnorm = _pload * _g->part() / ptotal;

    for (size_t l = 0; l < _g->nxyz(); l++)
    {
        power(l) *= _fnorm;
    }

}
