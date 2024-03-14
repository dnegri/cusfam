#include "SimonCPU.h"
#include "myblas.h"

extern "C" {
void updsfamxsec();
void runss(const bool& iternew, const bool& ifnodal, double& epsflx, double* flux, double& eigvl, double& reigvl, double& errflx);
}


SimonCPU::SimonCPU() {
    _eigv = 1.0;
}

SimonCPU::~SimonCPU() {

}

void SimonCPU::setBurnup(const char* dir_burn, const float& burnup, SteadyOption& option) {
    Simon::setBurnup(dir_burn, burnup, option);
    _iter_new = true;
}

void SimonCPU::initialize(const char* dbfile) {
    Simon::initialize(dbfile);
    _cmfd = new BICGCMFD(g(), x());
    _cmfd->init();
    cmfd().setNcmfd(5);
    cmfd().setEshift(0.04);
    updateCriteria(1.E-5);

    _ppr = new PinPower(g(), x());
}

void SimonCPU::updateCriteria(const float& crit_flux) {
    _crit_flux = crit_flux;
    _crit_xenon = crit_flux * 100.0;
}

void SimonCPU::runKeff(const int& nmaxout) {
    double errl2 = 0.0;
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
            d().eqxe(x().xsmica(), x().xsmicf(), _flux, _fnorm);
        }

        if (i > 3 && errl2 < 1.E-5) break;

        double reigv = 1. / _eigv;
        cmfd().updnodal(reigv, _flux, _jnet, _phis);
    }
}

void SimonCPU::runECP(const int& nmaxout, const double& eigvt) {
    double errl2 = 0.0;
    int nboiling = 0;

    cmfd().setNcmfd(5);
    cmfd().updpsi(_flux);

    float ppmd = _ppm;
    double eigvd = _eigv;
    double reigv = 1. / _eigv;
    for (int iout = 0; iout < nmaxout; iout++) {
        f().updatePPM(_ppm);
        d().updateH2ODensity(f().dm(), _ppm);
        x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());
        cmfd().upddtil();
        cmfd().setls(_eigv);
        cmfd().drive(_eigv, _flux, errl2);
        normalize();

        if (iout > 3 && errl2 < 1.E-5) break;

        if (errl2 < 1.E-4) {
            d().eqxe(x().xsmica(), x().xsmicf(), _flux, _fnorm);
            reigv = 1. / _eigv;
            cmfd().updnodal(reigv, _flux, _jnet, _phis);
        }

        //search critical
        f().updateTm(_power, nboiling);
        f().updateTf(_power, d().burn());

        double ppmn = updatePPM(iout == 0, eigvt, _ppm, ppmd, _eigv, eigvd);
        ppmd = _ppm;
        eigvd = _eigv;

        ppmn = ppmn;


        //nodal
    }

    //    for (int l = 0; l < _g->nxy(); ++l) {
    //        for (int ig = 0; ig < _g->ng(); ++ig) {
    //        }
    //    }
}

float SimonCPU::updatePPM(const bool& first, const double& eigvt, const float& ppm, const float& ppmd, const double& eigv, const double& eigvd) {
    double ppmn;

    if (first)
        ppmn = ppm + (_eigv - eigvt) * 1E5 / 10.0;
    else// if (abs(eigv - eigvd) > _crit_eigv)
        ppmn = (ppm - ppmd) / (eigv - eigvd) * (eigvt - eigv) + ppm;
    //else
    //	ppmn = ppm;

    if (ppmn > ppm + 300.0) {
        ppmn = ppm + 300.0;
    } else if (ppmn < ppm - 300.0) {
        ppmn = ppm - 300.0;
    }

    return ppmn;
}

void SimonCPU::runDepletion(const DepletionOption& option) {
    x().updateDepletionXS(f().dppm(), f().dtf(), f().dtm());
    d().pickData(x().xsmica(), x().xsmicf(), x().xsmic2n(), _flux, _fnorm);
    d().dep(option.tsec, option.xe, option.sm, option.xeamp, power());
}

void SimonCPU::runXenonTransient(const DepletionOption& option) {
    x().updateXeSmXS(f().dppm(), f().dtf(), f().dtm());
    d().dynxesm(option.tsec, option.xe, option.sm,  option.xeamp, x().xsmica(), x().xsmicf(), power(), _flux, _fnorm);
}

void SimonCPU::normalize() {
    double ptotal = 0;
    for (int l = 0; l < _g->nxyz(); l++) {
        power(l) = 0.0;
        for (int ig = 0; ig < _g->ng(); ig++) {
            power(l) += flux(ig, l) * x().xskf(ig, l);
        }
        power(l) *= _g->vol(l);
        ptotal += power(l);
    }

    _fnorm = _pload / ptotal;

    for (int l = 0; l < _g->nxyz(); l++) {
        power(l) *= _fnorm;
    }

}

void SimonCPU::runShapeMatch(const SteadyOption& condition) {

    runSteady(condition);

}

void SimonCPU::runSteady(const SteadyOption& condition) {
    int nboiling = 0;

    double eigvt = condition.eigvt;
    if(eigvt == 0.0) eigvt = _eigvc;

    _pload = _pload0 * std::max(condition.plevel, float(1.E-9)) * _g->part();

    //FIXME
    if (abs(condition.ppm - _ppm) > _crit_ppm) _ppm = condition.ppm;

    if(condition.b10a != 0) _d->updateB10Abundance(condition.b10a);

    if (abs(condition.tin - f().tin()) > _crit_tm) {
        f().updateTin(condition.tin);
        f().initDelta(condition.ppm);
    }
    if (condition.xenon == XEType::XE_NO) {
        d().multiplyDensity(XE45, 0.0);
    }
    if (condition.samarium == SMType::SM_NO) {
        d().multiplyDensity(SM49, 0.0);
    }


    float ppmd = _ppm;
    double eigvd = _eigv;
    bool updppm = true;
    bool updxs = true;
    bool updls = true;
    double errl2 = 1.0;
    double errl2d = 1.0;
    ShapeMatchOption shapeMatchOption = condition.shpmtch;
    for (int iout = 0; iout < condition.maxiter; iout++) {
        //FIXME
        if (updppm) {
            PLOG(plog::debug) << "Updating H2O and B10 number density";
            d().updateH2ODensity(f().dm(), _ppm);
            f().updatePPM(_ppm);
            updppm = false;
            updxs = true;
        }

        if (updxs) {
            PLOG(plog::debug) << "Updating cross-section and linear system";
            x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());
            x().updateDBC(d().dnst(), f().tf(), f().dppm(), f().dtf(), f().dtm());
            x().updateRodXS(r(), f().dppm(), f().dtf(), f().dtm());
            updxs = false;

            if (shapeMatchOption == ShapeMatchOption::SHAPE_MATCH) {
                double2 phi(_flux, _g->ng(), _g->nxyz());
                double2 cur(_jnet, _g->ng(), _g->nsurf());
                _shapeMatch->solve(eigvt, phi, cur, _powshp);
                updxs = true;

                PLOG(plog::debug) << "Updating Shape-Matching cross-sections";
            }

            if (shapeMatchOption != ShapeMatchOption::SHAPE_NO) x().updateShapeMatchXS(_shapeMatch->getShapeXS());
            cmfd().upddtil();
            updls = true;
        }

        if (updls) {
            cmfd().setls(_eigv);
            cmfd().updpsi(_flux);
            updls = false;
        }

        eigvd = _eigv;
        errl2d = errl2;

        cmfd().drive(_eigv, _flux, errl2);
        normalize();

        if (iout > 3 && errl2 < condition.epsiter && abs(_eigv - eigvd) < _crit_eigv) break;

        //FIXME uncomment
        if (errl2 > _crit_nodal) {
            PLOG(plog::debug) << "Updating nodal correction factor where errl2 is " << errl2 << " less than the criteria of " << _crit_nodal;
            cmfd().updnodal(_eigv, _flux, _jnet, _phis);
            updls = true;
        }


        //if (iout % 2 == 1 && errl2 < _crit_xenon && condition.xenon == XEType::XE_EQ) {
        if (condition.xenon == XEType::XE_EQ) {
            //if (condition.xenon == XEType::XE_EQ) {
            PLOG(plog::debug) << "Updating xenon density where errl2 is " << errl2 << " less than the criteria of " << _crit_xenon;

            x().updateXenonXS(f().dppm(), f().dtf(), f().dtm());
            d().eqxe(x().xsmica(), x().xsmicf(), _flux, _fnorm);
        }

        //search critical
        if (condition.feedtm) {
            PLOG(plog::debug) << "Updating moderator temp.";
            f().updateTm(_power, nboiling);
            updxs = true;
            updppm = true;
        }

        if (condition.feedtf) {
            PLOG(plog::debug) << "Updating fuel temp.";
            f().updateTf(_power, d().burn());
            updxs = true;
        }

        if (condition.searchOption == CriticalOption::CBC) {
            float ppmn = updatePPM(iout == 0, eigvt, _ppm, ppmd, _eigv, eigvd);
            ppmd = _ppm;
            _ppm = ppmn;
            updppm = true;

            PLOG(plog::debug) << "Updating boron concent. from " << ppmd << " to " << ppmn;
        }

    }
    myblas::multi(_g->ngxyz(), _fnorm, _flux, _flux);
    myblas::multi(_g->nsurf() * _g->ng(), _fnorm, _phis, _phis);
    myblas::multi(_g->nsurf() * _g->ng(), _fnorm, _jnet, _jnet);
    _fnorm = 1.0;
}

void SimonCPU::runSteadySfam(const SteadyOption& condition) {
    double errl2 = 1.0;
    int nboiling = 0;

    cmfd().updpsi(_flux);

    double eigvt = condition.eigvt;
    if(eigvt == 0.0) eigvt = _eigvc;

    _pload = _pload0 * std::max(condition.plevel, float(1.E-9)) * _g->part();

    if (abs(condition.tin - f().tin()) > _crit_tm) f().updateTin(condition.tin);
    if (abs(condition.ppm - _ppm) > _crit_ppm) _ppm = condition.ppm;
    if (condition.xenon == XEType::XE_NO) {
        d().multiplyDensity(XE45, 0.0);
    }

    if (condition.samarium == SMType::SM_NO) {
        d().multiplyDensity(SM49, 0.0);
    }


    float ppmd = _ppm;
    double eigvd = _eigv;
    bool updppm = true;
    bool updxs = true;
    bool updls = true;

    for (int iout = 0; iout < condition.maxiter; iout++) {
        //FIXME
        if (updppm) {
            PLOG(plog::debug) << "Updating H2O and B10 number density";
            f().updatePPM(_ppm);
            d().updateH2ODensity(f().dm(), _ppm);
            updppm = false;
            updxs = true;
        }

        if (updxs) {
            PLOG(plog::debug) << "Updating cross-section and linear system";
            x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());
            x().updateDBC(d().dnst(), f().tf(), f().dppm(), f().dtf(), f().dtm());
            x().updateRodXS(r(), f().dppm(), f().dtf(), f().dtm());
            //x().updateRodXSInTopRefl(g(), r(), f().dppm(), f().dtf(), f().dtm());
            updxs = false;
            updls = true;
        }

        double reigv = 1. / _eigv;
        double epsflux = 1.E-6;
        //FIXME
//		runss(_iter_new, true, epsflux, _flux, _eigv, reigv, errl2);

        _iter_new = false;

        normalize();

        if (iout > 3 && errl2 < condition.epsiter) break;

        //if (iout % 2 == 1 && errl2 < _crit_xenon && condition.xenon == XEType::XE_EQ) {
        //if (errl2 < _crit_xenon && condition.xenon == XEType::XE_EQ) {
        if (condition.xenon == XEType::XE_EQ) {
            PLOG(plog::debug) << "Updating xenon density where errl2 is " << errl2 << " less than the criteria of " << _crit_xenon;

            x().updateXenonXS(f().dppm(), f().dtf(), f().dtm());
            d().eqxe(x().xsmica(), x().xsmicf(), _flux, _fnorm);
        }

        //search critical
        if (condition.feedtm) {
            PLOG(plog::debug) << "Updating moderator temp.";
            f().updateTm(_power, nboiling);
            updxs = true;
            updppm = true;
        }

        if (condition.feedtf) {
            PLOG(plog::debug) << "Updating fuel temp.";
            f().updateTf(_power, d().burn());
            updxs = true;
        }

        if (condition.searchOption == CriticalOption::CBC) {
            float ppmn = updatePPM(iout == 0, eigvt, _ppm, ppmd, _eigv, eigvd);
            ppmd = _ppm;
            _ppm = ppmn;
            updppm = true;

            PLOG(plog::debug) << "Updating boron concent. from " << ppmd << " to " << ppmn;

        }

        eigvd = _eigv;
    }

    //myblas::multi(_g->ngxyz(), _fnorm, _flux, _flux);
    //_fnorm = 1.0;
}

void SimonCPU::runPinPower() {
    _ppr->calphicorn(_flux, _phis);
    _ppr->calhomo(_eigv, _flux, _phis, _jnet);
    _ppr->calpinpower();
    //_ppr->printPinPower(10);
    _ppr->applyFF(_ff_ptr, _d->burn());
    _fxy = _ppr->getFxy();
    _fr = _ppr->getFr();
    _fq = _ppr->getFq();
}

