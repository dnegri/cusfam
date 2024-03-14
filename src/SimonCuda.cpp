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
    _fcuda = new FeedbackCuda(*_gcuda, *_steamcuda);
    _dcuda = new DepletionCuda(*_gcuda);
    _cmfdcpu = new BICGCMFD(*_g, *_x);
    _cmfdcuda = new BICGCMFDCuda(*_gcuda, *_xcuda);

    _cmfdcuda->init();
    _cmfdcpu->init();

    _dcuda->init();
    _fcuda->allocate();

    _cmfdcuda->setNcmfd(3);
    _cmfdcuda->setEshift(0.04f);
    _cmfdcpu->setNcmfd(3);
    _cmfdcpu->setEshift(0.04f);

    checkCudaErrors(cudaMalloc((void**)&_flux_cuda, sizeof(double) * _g->ngxyz()));
    checkCudaErrors(cudaMalloc((void**)&_power_cuda, sizeof(double) * _g->nxyz()));
    checkCudaErrors(cudaMalloc((void**)&_jnet_cuda, sizeof(float) * _g->nsurf() * _g->ng()));


}

void SimonCuda::setBurnup(const float& burnup)
{
    Simon::setBurnup(burnup);

    _xcuda->copyXS(*_x);

    _dcuda->setDensity(_d->dnst());
    _dcuda->setBurnup(_d->burn());
    _dcuda->setH2ORatio(_d->h2on());

    _fcuda->copyFeedback(*_f);
    _fcuda->updatePressure(_press);
    _fcuda->updateTin(_tin);
    _fcuda->initDelta(_ppm);

    _xcuda->updateMacroXS(_dcuda->dnst());
    _xcuda->updateXS(_dcuda->dnst(), _fcuda->dppm(), _fcuda->dtf(), _fcuda->dtm());

    checkCudaErrors(cudaMemcpy(_flux_cuda, _flux, sizeof(double) * _g->ngxyz(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_power_cuda, _power, sizeof(double) * _g->nxyz(), cudaMemcpyHostToDevice));


    //memset(&_x->xsdf(0, 0), 0, sizeof(XS_PRECISION) * _g->ngxyz());
    //checkCudaErrors(cudaMemcpy(&_x->xsdf(0,0), _xcuda->xsdf(), sizeof(XS_PRECISION) * _g->ngxyz(), cudaMemcpyDeviceToHost));

}

void SimonCuda::runKeff(const int& nmaxout) {
    float errl2 = 0.0;
    int nboiling = 0;
    _cmfdcuda->setNcmfd(1);
    _cmfdcpu->setNcmfd(1);

    _cmfdcpu->updpsi(_flux);
    _cmfdcuda->updpsi(_flux_cuda);
    //checkCudaErrors(cudaMemcpy(&_cmfdcpu->psi(0), &_cmfdcuda->psi(0), sizeof(double) * _g->nxyz(), cudaMemcpyDeviceToHost));

    _ppm = 1000.0;
    _fcuda->updatePPM(_ppm);
    _dcuda->updateH2ODensity(_fcuda->dm(), _ppm);
    _xcuda->updateXS(_dcuda->dnst(), _fcuda->dppm(), _fcuda->dtf(), _fcuda->dtm());
    _cmfdcuda->upddtil();

    f().updatePPM(_ppm);
    d().updateH2ODensity(f().dm(), _ppm);
    x().updateXS(d().dnst(), f().dppm(), f().dtf(), f().dtm());
    _cmfdcpu->upddtil();

    _eigv = 1.0;
    _cmfdcuda->setls(_eigv);
    _cmfdcpu->setls(_eigv);
    //memset(&_cmfdcpu->diag(0, 0, 0), 0, sizeof(double) * _g->ng2() * _g->nxyz());
    //checkCudaErrors(cudaMemcpy(&_cmfdcpu->diag(0, 0, 0), &_cmfdcuda->diag(0, 0, 0), sizeof(double) * _g->ng2() * _g->nxyz(), cudaMemcpyDeviceToHost));

    _eigv = 1.0;
    _cmfdcpu->drive(_eigv, _flux, errl2);
    _eigv = 1.0;
    _cmfdcuda->drive(_eigv, _flux_cuda, errl2);
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
