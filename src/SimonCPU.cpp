#include "SimonCPU.h"
#include "myblas.h"

extern "C" {
	void updsfamxsec();
	void runss(const bool& iternew, const bool& ifnodal, double& epsflx, SOL_VAR* flux, double& eigvl, double& reigvl, double& errflx);
}



SimonCPU::SimonCPU() {

}

SimonCPU::~SimonCPU() {

}

void SimonCPU::initialize(const char* dbfile)
{
    Simon::initialize(dbfile);
    _cmfd = new BICGCMFD(g(), x());
    _cmfd->init();
    cmfd().setNcmfd(5);
    cmfd().setEshift(0.04);
	updateCriteria(1.E-5);
}

void SimonCPU::updateCriteria(const float& crit_flux) {
    _crit_flux = crit_flux;
    _crit_xenon = crit_flux*100.0;
    _crit_nodal = 1.0E-3;
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
	else if (abs(eigv - eigvd) > _crit_eigv)
		ppmn = (ppm - ppmd) / (eigv - eigvd) * (eigvt - eigv) + ppm;
	else
		ppmn = ppm;

    if(ppmn > ppm + 300.0) {
        ppmn = ppm + 300.0;
    } else if(ppmn < ppm - 300.0) {
        ppmn = ppm - 300.0;
    }

    return ppmn;
}

void SimonCPU::runDepletion(const DepletionOption& option) {
	printf("Updating DepletionXS.... ");
	x().updateDepletionXS(f().dppm(), f().dtf(), f().dtm());
	printf("Piking Depletion Data....  ");
    d().pickData(x().xsmica(), x().xsmicf(), x().xsmic2n(), _flux, _fnorm);
	printf("Run Depletion.... ");
    d().dep(option.tsec, option.xe, option.sm, power());
	printf("DONE!\n");
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

    _fnorm = _pload / ptotal;

    for (int l = 0; l < _g->nxyz(); l++)
    {
        power(l) *= _fnorm;
    }

}

void SimonCPU::runSteady(const SteadyOption& condition) {
    float errl2 = 1.0;
    int nboiling = 0;

    cmfd().updpsi(_flux);

	_pload = _pload0*std::max(condition.plevel, float(1.E-9))*_g->part();

    if(condition.tin != 0) f().updateTin(condition.tin);
	if(condition.ppm != 0) _ppm = condition.ppm;
	if (condition.xenon == XEType::XE_NO) {
		d().multiplyDensity(XE45, 0.0);
	}


	float ppmd = _ppm;
	double eigvd = _eigv;
	bool updppm = true;
	bool updxs = true;
	bool updls = true;
    for (int iout = 0; iout < condition.maxiter; iout++)
    {
		if (updppm) {
			PLOG(plog::debug) << "Updating H2O and B10 number density";
			printf("B10 and H2O Update\n");
			f().updatePPM(_ppm);
			d().updateH2ODensity(f().dm(), _ppm);
			updppm = false;
			updxs = true;
		}

		if (updxs) {
			PLOG(plog::debug) << "Updating cross-section and linear system";
			printf("XS Update\n");
			x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());
			printf("XS Rod Update\n");
			x().updateRodXS(r(), f().dppm(), f().dtf(), f().dtm());
			printf("DTIL Update\n");
			cmfd().upddtil();
			printf("DTIL Updated\n");
			updxs = false;
			updls = true;
		}

		if (errl2 < _crit_nodal) {
			PLOG(plog::debug) << "Updating nodal correction factor where errl2 is " << errl2 << " less than the criteria of " << _crit_nodal;
			printf("Nodal Calculation\n");
			cmfd().updnodal(_eigv, _flux, _jnet);
			updls = true;
		}

		if (updls) {
			printf("LS Update\n");
			cmfd().setls(_eigv);
			printf("LS Updated\n");
			updls = false;
		}

		printf("CMFD CALC\n");
		cmfd().drive(_eigv, _flux, errl2);
		printf("CMFD CALCed\n");
		normalize();

        if (iout > 3 && errl2 < _crit_flux) break;

        //if (iout % 2 == 1 && errl2 < _crit_xenon && condition.xenon == XEType::XE_EQ) {
		if (errl2 < _crit_xenon && condition.xenon == XEType::XE_EQ) {
			PLOG(plog::debug) << "Updating xenon density where errl2 is " << errl2 << " less than the criteria of " << _crit_xenon;
			printf("Xenon Update\n");

			x().updateXenonXS(f().dppm(), f().dtf(), f().dtm());
            d().eqxe(x().xsmica(), x().xsmicf(), _flux, _fnorm);
        }

        //search critical
		if (condition.feedtm) {
			PLOG(plog::debug) << "Updating moderator temp.";
			printf("TM Update\n");
			f().updateTm(_power, nboiling);
			updxs = true;
			updppm = true;
		}

		if (condition.feedtf) {
			PLOG(plog::debug) << "Updating fuel temp.";
			printf("TF Update\n");
			f().updateTf(_power, d().burn());
			updxs = true;
		}

        if(condition.searchOption == CriticalOption::CBC) {
            float ppmn = updatePPM(iout == 0, condition.eigvt, _ppm, ppmd, _eigv, eigvd);
            ppmd = _ppm;
            _ppm = ppmn;
			updppm = true;

			PLOG(plog::debug) << "Updating boron concent. from " << ppmd << " to " << ppmn;
			printf("PPM Update\n");

        }

        eigvd = _eigv;
    }
	myblas::multi(_g->ngxyz(), _fnorm, _flux, _flux);
	_fnorm = 1.0;

}

void SimonCPU::runSteadySfam(const SteadyOption& condition) {
	double errl2 = 1.0;
	int nboiling = 0;

	cmfd().updpsi(_flux);

	_pload = _pload0 * std::max(condition.plevel, float(1.E-9))*_g->part();

	if (condition.tin != 0) f().updateTin(condition.tin);
	if (condition.ppm != 0) _ppm = condition.ppm;
	if (condition.xenon == XEType::XE_NO) {
		d().multiplyDensity(XE45, 0.0);
	}


	float ppmd = _ppm;
	double eigvd = _eigv;
	bool updppm = true;
	bool updxs = true;
	bool updls = true;
	for (int iout = 0; iout < condition.maxiter; iout++)
	{
		if (updppm) {
			PLOG(plog::debug) << "Updating H2O and B10 number density";
			printf("B10 and H2O Update\n");
			f().updatePPM(_ppm);
			d().updateH2ODensity(f().dm(), _ppm);
			updppm = false;
			updxs = true;
		}

		if (updxs) {
			PLOG(plog::debug) << "Updating cross-section and linear system";
			printf("XS Update\n");
			x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());
			printf("XS Rod Update\n");
			x().updateRodXS(r(), f().dppm(), f().dtf(), f().dtm());
			printf("DTIL Update\n");
			//updsfamxsec();
			printf("DTIL Updated\n");
			updxs = false;
			updls = true;
		}

		double reigv = 1. / _eigv;
		double epsflux = 1.E-6;
		runss(iout == 0, iout != 0, epsflux, _flux, _eigv, reigv, errl2);

		normalize();

		double sum = 0.0;
		accumulate(_flux, _flux + 12532, sum);
		printf("FLUX : %12.5E, FNORM: %12.5E\n", sum/12532.0, _fnorm);

		if (iout > 3 && errl2 < _crit_flux) break;

		//if (iout % 2 == 1 && errl2 < _crit_xenon && condition.xenon == XEType::XE_EQ) {
		//if (errl2 < _crit_xenon && condition.xenon == XEType::XE_EQ) {
		if (condition.xenon == XEType::XE_EQ) {
			PLOG(plog::debug) << "Updating xenon density where errl2 is " << errl2 << " less than the criteria of " << _crit_xenon;
			printf("Xenon Update\n");

			x().updateXenonXS(f().dppm(), f().dtf(), f().dtm());
			d().eqxe(x().xsmica(), x().xsmicf(), _flux, _fnorm);
		}

		//search critical
		if (condition.feedtm) {
			PLOG(plog::debug) << "Updating moderator temp.";
			printf("TM Update\n");
			f().updateTm(_power, nboiling);
			updxs = true;
			updppm = true;
		}

		if (condition.feedtf) {
			PLOG(plog::debug) << "Updating fuel temp.";
			printf("TF Update\n");
			f().updateTf(_power, d().burn());
			updxs = true;
		}

		if (condition.searchOption == CriticalOption::CBC) {
			float ppmn = updatePPM(iout == 0, condition.eigvt, _ppm, ppmd, _eigv, eigvd);
			ppmd = _ppm;
			_ppm = ppmn;
			updppm = true;

			PLOG(plog::debug) << "Updating boron concent. from " << ppmd << " to " << ppmn;
			printf("PPM Update\n");

		}

		eigvd = _eigv;
	}

	myblas::multi(_g->ngxyz(), _fnorm, _flux, _flux);
	_fnorm = 1.0;
}

