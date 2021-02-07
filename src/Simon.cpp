#include "Simon.h"

extern "C" {
void opendb(int* length, const char* file);
void readDimension(int* ng, int* nxy, int* nz, int* nx, int* ny, int* nsurf);
void readIndex(int* nx, int* ny, int* nxy, int* nz, int* nxs, int* nxe, int* nys, int* nye, int* ijtol, int* neibr,
               float* hmesh);
void readBoundary(int* symopt, int* symang, float* albedo);
void readNXNY(const int* nx, const int* ny, float* val);
void readNXYZ(const int* nxyz, float* val);
//void readNXYZ8(const int* nxyz, double* val);
//void readNXYZI(const int* nxyz, int* val);

void readStep(float* bucyc, float* buavg, float* efpd, double* ppm, double* eigv, double* power, double* fnorm);
void readXS(const int* niso, float* xs);
void readXSS(const int* niso, float* xs);
void readXSD(const int* niso, float* xs);
void readXSSD(const int* niso, float* xs);
void readXSDTM(const int* niso, float* xs);
void readXSSDTM(const int* niso, float* xs);
void readDensity(const int* niso, float* dnst);

}

Simon::Simon() {
    _epsbu = 5.0;
}

Simon::~Simon() {

}

void Simon::initialize(const char* dbfile) {


    int length = strlen(dbfile);
    opendb(&length, dbfile);

    int ng;
    int nx;
    int ny;
    int nz;
    int nxy;
    int nxyz;
    int nsurf;

    readDimension(&ng, &nxy, &nz, &nx, &ny, &nsurf);
    nxyz = nxy * nz;
    nsurf = nsurf * nz + (nz + 1) * nxy;


    _g = new Geometry();

    int* nxs = new int[ny];
    int* nxe = new int[ny];
    int* nys = new int[nx];
    int* nye = new int[nx];
    int* ijtol = new int[nx * ny];
    int* neibr = new int[NEWS * nxy];
    float* hmesh = new float[NDIRMAX * nxyz];

    int symopt;
    int symang;
    float albedo[6];

    readIndex(&nx, &ny, &nxy, &nz, nxs, nxe, nys, nye, ijtol, neibr, hmesh);
    readBoundary(&symopt, &symang, albedo);

    _g->initDimension(&ng, &nxy, &nz, &nx, &ny, &nsurf);
    _g->initIndex(nxs, nxe, nys, nye, ijtol, neibr, hmesh);
    _g->setBoudnaryCondition(&symopt, &symang, albedo);

    _x = new CrossSection(ng, NISO, NFIS, NNIS, NPTM, nxyz);

    _d = new DepletionChain(*_g);

    _steam = new SteamTable();
    _f = new Feedback(*_g, *_steam);

    cmfd = new BICGCMFD(*_g, *_x);
    cmfd->setNcmfd(5);
    cmfd->setEshift(0.04);


    readNXYZ(&nxy, &(_f->chflow(0)));
    readNXYZ(&(_g->nxyz()), &(_f->ppm0(0)));
    readNXYZ(&(_g->nxyz()), &(_f->stf0(0)));
    readNXYZ(&(_g->nxyz()), &(_f->tm0(0)));
    readNXYZ(&(_g->nxyz()), &(_f->dm0(0)));
    readNXYZ(&(_g->nxyz()), &(_d->h2on(0)));
    int one = 1;
//    readNXYZI(&nxyz, &(_f->fueltype(0)));
    readNXYZ(&nxy, &(_f->frodn(0)));
//    readNXYZI(&one, &(_f->nft()));

    _f->initTFTable(_f->nft());
//    readNXYZI(&(_f->nft()), &(_f->ntfbu(0)));
//    readNXYZI(&(_f->nft()), &(_f->ntfpow(0)));
    int size = TF_POINT * _f->nft();
    readNXYZ(&size, &(_f->tfbu(0,0)));
    readNXYZ(&size, &(_f->tfpow(0, 0)));
    size = size * TF_POINT;
    readNXYZ(&size, &(_f->tftable(0,0,0)));


    _power = new float[_g->nxyz()]{};
    _flux = new double[_g->ngxyz()]{};
    _jnet = new double[_g->nsurf() * _g->ng()]{};

    //readBurnupList
    _nstep = 3;
    _bucyc = new float[_nstep] {0.0};

    delete[] nxs;
    delete[] nxe;
    delete[] nys;
    delete[] nye;
    delete[] ijtol;
    delete[] neibr;
    delete[] hmesh;
}

void Simon::setBurnup(const float& burnup) {

    int i = _nstep-1;
    for (; i >= 0; --i) {
        if(_bucyc[i] < burnup) break;
    }

    if(i == -1) i=0;


    //skip (i-1) burnup data
    //skip

    //read (i)-th burnup data
    float bucyc, buavg, efpd;
    readStep(&bucyc, &buavg, &efpd, &_ppm, &_eigv, &_pload, &_fnorm);

    _reigv = 1. / _eigv;

    readDensity(&NISO, &(_d->dnst(0, 0)));
    readNXYZ(&(_g->nxyz()), &(_d->burn(0)));
    readNXYZ(&(_g->nxyz()), _power);
//    readNXYZ8(&(_g->ngxyz()), _flux);
    readNXYZ(&(_g->nxyz()), &(_f->tf(0)));
    readNXYZ(&(_g->nxyz()), &(_f->tm(0)));
    readNXYZ(&(_g->nxyz()), &(_f->dm(0)));

    readXS(&NISO, &(_x->xsmicd0(0, 0, 0)));
    readXS(&NISO, &(_x->xsmica0(0, 0, 0)));
    readXS(&NISO, &(_x->xsmicf0(0, 0, 0)));
    readXS(&NISO, &(_x->xsmicn0(0, 0, 0)));
    readXS(&NISO, &(_x->xsmick0(0, 0, 0)));
    readXSS(&NISO, &(_x->xsmics0(0, 0, 0, 0)));

    readXSD(&NISO, &(_x->xdpmicd(0, 0, 0)));
    readXSD(&NISO, &(_x->xdpmica(0, 0, 0)));
    readXSD(&NISO, &(_x->xdpmicf(0, 0, 0)));
    readXSD(&NISO, &(_x->xdpmicn(0, 0, 0)));
    readXSD(&NISO, &(_x->xdpmick(0, 0, 0)));
    readXSSD(&NISO, &(_x->xdpmics(0, 0, 0, 0)));

    readXSD(&NISO, &(_x->xdfmicd(0, 0, 0)));
    readXSD(&NISO, &(_x->xdfmica(0, 0, 0)));
    readXSD(&NISO, &(_x->xdfmicf(0, 0, 0)));
    readXSD(&NISO, &(_x->xdfmicn(0, 0, 0)));
    readXSD(&NISO, &(_x->xdfmick(0, 0, 0)));
    readXSSD(&NISO, &(_x->xdfmics(0, 0, 0, 0)));

    readXSD(&NISO, &(_x->xddmicd(0, 0, 0)));
    readXSD(&NISO, &(_x->xddmica(0, 0, 0)));
    readXSD(&NISO, &(_x->xddmicf(0, 0, 0)));
    readXSD(&NISO, &(_x->xddmicn(0, 0, 0)));
    readXSD(&NISO, &(_x->xddmick(0, 0, 0)));
    readXSSD(&NISO,&(_x->xddmics(0, 0, 0, 0)));


    readXSDTM (&NISO, &(_x->xdmmicd(0, 0, 0, 0)));
    readXSDTM(&NISO, &(_x->xdmmica(0, 0, 0, 0)));
    readXSDTM(&NISO, &(_x->xdmmicf(0, 0, 0, 0)));
    readXSDTM(&NISO, &(_x->xdmmicn(0, 0, 0, 0)));
    readXSDTM(&NISO, &(_x->xdmmick(0, 0, 0, 0)));
    readXSSDTM(&NISO, &(_x->xdmmics(0, 0, 0, 0, 0)));


    float dburn = burnup-bucyc;
    if(dburn > _epsbu) runDepletion(dburn);
    _x->updateMacroXS(&(_d->dnst(0, 0)));


    _f->initDelta(_ppm);
    _x->updateXS(&(_d->dnst(0, 0)), &(_f->dppm(0)), &(_f->dtf(0)), &(_f->dtm(0)));
}

void Simon::runStatic(const int& nmaxout) {
    float errl2 = 0.0;

    cmfd->setEshift(0.04);
    cmfd->updpsi(_flux);

    for (size_t iout = 0; iout < nmaxout; iout++)
    {
        _x->updateXS(_d->dnst(), _f->dppm(), _f->dtf(), _f->dtm());
        cmfd->upddtil();
        cmfd->setls(_eigv);
        cmfd->drive(_reigv, _flux, errl2);
        normalize();
        //search critical
        _f->updateTf(_power, _d->burn());
        int nboiling = 0;
        _f->updateTm(_power, nboiling);

        _eigv = 1. / _reigv;
        if (errl2 < 1.E-5) break;
    }
}

void Simon::runDepletion(const float& dburn) {

}

void Simon::runXenonTransient() {

}

void Simon::normalize()
{
    double ptotal = 0;
    for (size_t l = 0; l < _g->nxyz(); l++)
    {
        power(l) = 0.0;
        for (size_t ig = 0; ig < _g->ng(); ig++)
        {
            power(l) += flux(ig, l) * _x->xskf(ig, l);
        }
        ptotal += power(l);
    }

    _fnorm = _pload * 0.25 / ptotal;

    for (size_t l = 0; l < _g->nxyz(); l++)
    {
        power(l) *= _fnorm;
        for (size_t ig = 0; ig < _g->ng(); ig++)
        {
            flux(ig, l) *= _fnorm;
        }
    }

}
