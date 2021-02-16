#include "SimonCuda.h"
#include "myblas.h"


SimonCuda::SimonCuda() {
}

SimonCuda::~SimonCuda() {

}

void SimonCuda::initialize(const char* dbfile)
{
    Simon::initialize(dbfile);
    _gcuda = new GeometryCuda(*_g);
    _xcuda = new CrossSectionCuda(*_x);
    _steamcuda = new SteamTableCuda(*_steam);
    _fcuda = new FeedbackCuda(*_gcuda, *_steamcuda, *_f);
    _dcuda = new DepletionCuda(*_gcuda);
    _dcuda->init();

    _cmfdcuda = new BICGCMFDCuda(*_gcuda, *_xcuda);
    _cmfdcuda->init();
    _cmfdcuda->setNcmfd(3);
    _cmfdcuda->setEshift(0.04f);
}

void SimonCuda::setBurnup(const float& burnup)
{
    Simon::setBurnup(burnup);
    _dcuda->setDensity(_d->dnst());
    _dcuda->setBurnup(_d->burn());
    _dcuda->setH2ORatio(_d->h2on());

    _fcuda->copyFeedback(*_f);
    _fcuda->updatePressure(_press);
    _fcuda->updateTin(_tin);
    _fcuda->initDelta(_ppm);

    _xcuda->copyXS(*_x);
    _xcuda->updateMacroXS(_dcuda->dnst());
    _xcuda->updateXS(_dcuda->dnst(), _fcuda->dppm(), _fcuda->dtf(), _fcuda->dtm());
}

void SimonCuda::runKeff(const int& nmaxout) {
    float errl2 = 0.0;
    int nboiling = 0;
    _cmfdcuda->setNcmfd(nmaxout);

    _cmfdcuda->updpsi(_flux);

    _ppm = 100.0;
    _fcuda->updatePPM(_ppm);
    _dcuda->updateH2ODensity(_fcuda->dm(), _ppm);
    _xcuda->updateXS(_dcuda->dnst(), _fcuda->dppm(), _fcuda->dtf(), _fcuda->dtm());
    _cmfdcuda->upddtil();
    _cmfdcuda->setls(_eigv);
    _cmfdcuda->drive(_eigv, _flux, errl2);

    _ppm = 1000.0;
    _fcuda->updatePPM(_ppm);
    _dcuda->updateH2ODensity(_fcuda->dm(), _ppm);
    _xcuda->updateXS(_dcuda->dnst(), _fcuda->dppm(), _fcuda->dtf(), _fcuda->dtm());
    _cmfdcuda->upddtil();
    _cmfdcuda->setls(_eigv);
    _cmfdcuda->drive(_eigv, _flux, errl2);
    exit(0);
    normalize();
}

void SimonCuda::runECP(const int& nmaxout, const double& eigvt) {
    float errl2 = 0.0;
    int nboiling = 0;

    _cmfdcuda->setNcmfd(3);
    _cmfdcuda->updpsi(_flux);

    float ppmd = _ppm;
    double eigvd = _eigv;

    for (size_t iout = 0; iout < nmaxout; iout++)
    {
        _fcuda->updatePPM(_ppm);
        _dcuda->updateH2ODensity(_fcuda->dm(), _ppm);
        _xcuda->updateXS(_dcuda->dnst(), _fcuda->dppm(), _fcuda->dtf(), _fcuda->dtm());
        _cmfdcuda->upddtil();
        _cmfdcuda->setls(_eigv);
        _cmfdcuda->drive(_eigv, _flux, errl2);
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
        _fcuda->updateTm(_power, nboiling);
        _fcuda->updateTf(_power, _dcuda->burn());
        _dcuda->eqxe(_xcuda->xsmica(), _xcuda->xsmicf(), _flux, _fnorm);
    }
}

void SimonCuda::runDepletion(const float& dburn) {

    _dcuda->pickData(_xcuda->xsmica(), _xcuda->xsmicf(), _xcuda->xsmic2n(), _flux, _fnorm);
    _dcuda->dep(116748.0);
}

void SimonCuda::runXenonTransient() {

}

void SimonCuda::normalize()
{
    double ptotal = 0;
    for (size_t l = 0; l < _g->nxyz(); l++)
    {
        power(l) = 0.0;
        for (size_t ig = 0; ig < _g->ng(); ig++)
        {
            power(l) += flux(ig, l) * _xcuda->xskf(ig, l);
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
