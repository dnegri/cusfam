#include "Simon.h"
#include "myblas.h"

extern "C" {
    void opendb(int* length, const char* file);
    void readDimension(int* ng, int* nxy, int* nz, int* nx, int* ny, int* nsurf);
    void readIndex(int* nx, int* ny, int* nxy, int* nz, int* nxs, int* nxe, int* nys, int* nye, int* ijtol, int* neibr,
                   float* hmesh);
    void readBoundary(int* symopt, int* symang, float* albedo);
    void readNXNY(const int* nx, const int* ny, float* val);
    void readNXYZ(const int* nxyz, float* val);
    void readNXYZ8(const int* nxyz, double* val);
    void readNXYZI(const int* nxyz, int* val);

    void readStep(float* bucyc, float* buavg, float* efpd, double* eigv, double* power, double* fnorm);
    void readConstantF(const int& n, float* data);
    void readConstantD(const int& n, double* data);
    void readConstantI(const int& n, int* data);

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
    _g->setBoundaryCondition(&symopt, &symang, albedo);

    _x = new CrossSection(ng, nxyz);

    _d = new Depletion(*_g);
    _d->init();

	float b10ap;
	readConstantF(1, &b10ap);
	_d->updateB10Abundance(b10ap);
	
	readConstantF(1, &_d->totmass());
	readNXYZ(&nxyz, _d->buconf());

    _steam = new SteamTable();
    _steam->setPressure(155.13);

    _f = new Feedback(*_g, *_steam);
    _f->allocate();

    readNXYZ(&nxy, &(_f->chflow(0)));
    readNXYZ(&(_g->nxyz()), &(_f->ppm0(0)));
    readNXYZ(&(_g->nxyz()), &(_f->stf0(0)));
    readNXYZ(&(_g->nxyz()), &(_f->tm0(0)));
    readNXYZ(&(_g->nxyz()), &(_f->dm0(0)));
    readNXYZ(&(_g->nxyz()), &(_d->h2on(0)));

    int one = 1;
    readNXYZI(&nxyz, &(_f->fueltype(0)));
    readNXYZ(&nxy, &(_f->frodn(0)));
    readNXYZI(&one, &(_f->nft()));

    _f->initTFTable(_f->nft());
    readNXYZI(&(_f->nft()), &(_f->ntfbu(0)));
    readNXYZI(&(_f->nft()), &(_f->ntfpow(0)));
    int size = TF_POINT * _f->nft();
    readNXYZ(&size, &(_f->tfbu(0,0)));
    readNXYZ(&size, &(_f->tfpow(0, 0)));
    size = size * TF_POINT;
    readNXYZ(&size, &(_f->tftable(0,0,0)));


    _power = new float[_g->nxyz()]{};
    _flux = new SOL_VAR[_g->ngxyz()]{};
    _jnet = new SOL_VAR[_g->nsurf() * _g->ng()]{};

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
    readStep(&bucyc, &buavg, &efpd, &_eigv, &_pload, &_fnorm);

    _reigv = 1. / _eigv;

    readDensity(&NISO, &(_d->dnst(0, 0)));
    readNXYZ(&(_g->nxyz()), &(_d->burn(0)));
    readNXYZ(&(_g->nxyz()), _power);

    double* temp = new double[_g->ngxyz()];
    readNXYZ8(&(_g->ngxyz()), temp);
    std::copy_n(temp, _g->ngxyz(), _flux);
    delete[] temp;

    float data[100];
    readConstantF(3, data);
    _press = data[0];
    _tin   = data[1];
    _ppm   = data[2];

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

    _x->updateMacroXS(&(_d->dnst(0, 0)));

    _f->updatePressure(_press);
    _f->updateTin(_tin);
    _f->initDelta(_ppm);
    _x->updateXS(&(_d->dnst(0, 0)), &(_f->dppm(0)), &(_f->dtf(0)), &(_f->dtm(0)));

}

void Simon::setFeedbackOption(bool feed_tf, bool feed_tm)
{
    _feed_tf = feed_tf;
    _feed_tm = feed_tm;
}

__host__ void Simon::print(Geometry& g, CrossSection& x, Feedback& f, Depletion& d)
{
    for (int l = 0; l < g.nxyz(); l++)
    {
        printf("%e %e %e %e %e %e %e %e \n", x.xsdf(0,l), x.xstf(0, l), x.chif(0, l), x.xsnf(0, l), x.xssf(0, 1, l), f.tf(l), f.tm(l), d.dnst(2,l));
    }
    
}
