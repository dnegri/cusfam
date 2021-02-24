#include "SimonCPU.h"
#include "myblas.h"


SimonCPU::SimonCPU() {

}

SimonCPU::~SimonCPU() {

}

void SimonCPU::initialize(const char* dbfile)
{
    updateCriteria(1.E-5);
    Simon::initialize(dbfile);
    _cmfd = new BICGCMFD(g(), x());
    _cmfd->init();
    cmfd().setNcmfd(4);
    cmfd().setEshift(0.1);

}

void SimonCPU::updateCriteria(const float& crit_flux) {
    _crit_flux = crit_flux;
    _crit_xenon = crit_flux*10.0;
    _crit_nodal = crit_flux*10.0;
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
    for (int iout = 0; iout < nmaxout; iout++)
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

        double ppmn = updatePPM(iout==0, eigvt, _ppm, ppmd, _eigv, eigvd);
        ppmd = _ppm;
        eigvd = _eigv;

        ppmn = ppmn;

        printf("CHANGE PPM : %.2f --> %.2f\n", ppmd, _ppm);

        //nodal
    }

//    for (int l = 0; l < _g->nxy(); ++l) {
//        for (int ig = 0; ig < _g->ng(); ++ig) {
//            printf("FLUX : %e\n", flux(ig,l)*_fnorm);
//        }
//    }
}

float SimonCPU::updatePPM(const bool& first, const double& eigvt, const float& ppm, const float& ppmd, const double& eigv, const double& eigvd) {
    double ppmn;

    if (first)
        ppmn = ppm + (_eigv - eigvt) * 1E5 / 10.0;
    else
        ppmn = (ppm - ppmd) / (eigv - eigvd) * (eigvt - eigv) + ppm;

    if(ppmn > ppm + 300.0) {
        ppmn = ppm + 300.0;
    } else if(ppmn < ppm - 300.0) {
        ppmn = ppm - 300.0;
    }

    return ppmn;
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
    for (int l = 0; l < _g->nxyz(); l++)
    {
        power(l) = 0.0;
        for (int ig = 0; ig < _g->ng(); ig++)
        {
            power(l) += flux(ig, l) * x().xskf(ig, l);
        }
        power(l) *= _g->vol(l);
        ptotal += power(l);
    }

    _fnorm = _pload * _g->part() / ptotal;

    for (int l = 0; l < _g->nxyz(); l++)
    {
        power(l) *= _fnorm;
    }

}

void SimonCPU::runSteady(const SteadyOption& condition) {
    float errl2 = 0.0;
    int nboiling = 0;

    cmfd().updpsi(_flux);

    float ppmd = _ppm;
    double eigvd = _eigv;

    if(condition.tin != 0) f().updateTin(condition.tin);

    for (int iout = 0; iout < condition.maxiter; iout++)
    {
        f().updatePPM(_ppm);
        d().updateH2ODensity(f().dm(), _ppm);
        x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());
        cmfd().upddtil();
        cmfd().setls(_eigv);
        cmfd().drive(_eigv, _flux, errl2);
        normalize();

        if (iout > 3 && errl2 < _crit_flux) break;

        if (errl2 < _crit_xenon && condition.xenon == XEType::XE_EQ) {
            d().eqxe(x().xsmica(), x().xsmicf(), _flux, _fnorm);
        }
        if (errl2 < _crit_nodal) {
            cmfd().updnodal(_eigv, _flux, _jnet);
        }

        //search critical
        if(condition.feedtm) f().updateTm(_power, nboiling);
        if(condition.feedtf) f().updateTf(_power, d().burn());

        if(condition.searchOption == SearchOption::CBC) {
            double ppmn = updatePPM(iout == 0, condition.eigvt, _ppm, ppmd, _eigv, eigvd);
            ppmd = _ppm;
            _ppm = ppmn;
        }

        eigvd = _eigv;
    }
}
