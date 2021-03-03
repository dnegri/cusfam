#include "Simon.h"
#include "myblas.h"

extern "C" {
	void opendb(int* length, const char* file);
	void readTableSet(int* length, const char* file, const int* ncomp, char** compnames);
	void closedb();
	void readDimension(int* ng, int* nxy, int* nz, int* nx, int* ny, int* nsurf);
	void readIndex(int* nx, int* ny, int* nxy, int* nz, int* nxs, int* nxe, int* nys, int* nye, int* ijtol, int* neibr,
		float* hmesh);
	void readComposition(const int* nxy, const int*nz, int* ncomp, char** names, int* comps);

	void readBoundary(int* symopt, int* symang, float* albedo);
	void readNXNY(const int* nx, const int* ny, float* val);
	void readNXYZ(const int* nxyz, float* val);
	void readNXYZ8(const int* nxyz, double* val);
	void readNXYZI(const int* nxyz, int* val);

	void readStep(float* plevel, float* bucyc, float* buavg, float* efpd, double* eigv, double* fnorm);
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


	void calculateReference(const int& icomp, const float& burn, const float* xsmicd, const float* xsmica, const float* xsmicn,
		const float* xsmicf, const float* xsmick, const float* xsmics, const float* xsmic2n, const float* xehfp);

	void calculateVariation(const int& icomp, const float& burn,
		const float*xdpmicn, const float*xdfmicn, const float* xdmmicn, const float* xddmicn,
		const float* xdpmicf, const float* xdfmicf, const float* xdmmicf, const float* xddmicf,
		const float* xdpmica, const float* xdfmica, const float* xdmmica, const float* xddmica,
		const float* xdpmicd, const float* xdfmicd, const float* xdmmicd, const float* xddmicd,
		const float* xdpmics, const float* xdfmics, const float* xdmmics, const float* xddmics);

}

Simon::Simon() {
	_epsbu = 5.0;
}

Simon::~Simon() {

}

void Simon::initialize(const char* dbfile) {


	int length = strlen(dbfile);
	opendb(&length, dbfile);

	int one = 1;

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

	int ncomp = 0;
	int* comp = new int[nxyz];

	readComposition(&nxy, &nz, &_g->ncomp(), _g->compnames(), comp);

	_x = new CrossSection(ng, nxyz);

	_d = new Depletion(*_g);
	_d->init();

	float b10ap;
	readConstantF(1, &b10ap);
	_d->updateB10Abundance(b10ap);

	readConstantF(1, &_pload0);
	_pload = _pload0 * _g->part();


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

	readNXYZI(&nxyz, &(_f->fueltype(0)));
	readNXYZ(&nxy, &(_f->frodn(0)));
	readNXYZI(&one, &(_f->nft()));

	_f->initTFTable(_f->nft());
	readNXYZI(&(_f->nft()), &(_f->ntfbu(0)));
	readNXYZI(&(_f->nft()), &(_f->ntfpow(0)));
	int size = TF_POINT * _f->nft();
	readNXYZ(&size, &(_f->tfbu(0, 0)));
	readNXYZ(&size, &(_f->tfpow(0, 0)));
	size = size * TF_POINT;
	readNXYZ(&size, &(_f->tftable(0, 0, 0)));


	_power = new float[_g->nxyz()]{};
	_flux = new SOL_VAR[_g->ngxyz()]{};
	_jnet = new SOL_VAR[_g->nsurf() * _g->ng()]{};

	//readBurnupList
	_nstep = 18;

	_bucyc = new float[_nstep] {0, 50, 150, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 13650};

	closedb();

	delete[] nxs;
	delete[] nxe;
	delete[] nys;
	delete[] nye;
	delete[] ijtol;
	delete[] neibr;
	delete[] hmesh;
}

void Simon::readTableSet(const char * tsetfile)
{
	int length = strlen(tsetfile);
	::readTableSet(&length, tsetfile, &_g->ncomp(), _g->compnames());
}

void Simon::updateBurnup()
{
	for (int l = 0; l < _g->nxyz(); l++)
	{
		calculateReference(_g->comp(l), _d->burn(l), 
			_x->xsmicd0(l), _x->xsmica0(l), _x->xsmicn0(l), _x->xsmicf0(l), 
			_x->xsmick0(l), _x->xsmics0(l), _x->xsmic2n(l), _x->xehfp(l));

		calculateVariation(_g->comp(l), _d->burn(l), 
			_x->xdpmicn(l), _x->xdfmicn(l), _x->xdmmicn(l), _x->xddmicn(l),
			_x->xdpmicf(l), _x->xdfmicf(l), _x->xdmmicf(l), _x->xddmicf(l),
			_x->xdpmica(l), _x->xdfmica(l), _x->xdmmica(l), _x->xddmica(l),
			_x->xdpmicd(l), _x->xdfmicd(l), _x->xdmmicd(l), _x->xddmicd(l),
			_x->xdpmics(l), _x->xdfmics(l), _x->xdmmics(l), _x->xddmics(l));
	}
}

void Simon::setBurnup(const float& burnup) {

	int i = 0;
	for (; i < _nstep; ++i) {
		if (burnup - 10.0 < _bucyc[i]) break;
	}

	char dbfile[19];
	int intbu = round(_bucyc[i]);
	sprintf(dbfile, "../run/%05d.simon", intbu);
	printf("Started reading burn file : %s\n", dbfile);

	int length = strlen(dbfile);
	opendb(&length, dbfile);

	float bucyc, buavg, efpd;
	readStep(&_pload, &bucyc, &buavg, &efpd, &_eigv, &_fnorm);

	_reigv = 1. / _eigv;


	readDensity(&NISO, _d->dnst_new());

	if (burnup == 0.0) {
		std::copy_n(_d->dnst_new(), NISO * _g->nxyz(), _d->dnst());
	}


	readNXYZ(&(_g->nxyz()), &(_d->burn(0)));
	readNXYZ(&(_g->nxyz()), _power);

	double* temp = new double[_g->ngxyz()];
	readNXYZ8(&(_g->ngxyz()), temp);
	std::copy_n(temp, _g->ngxyz(), _flux);
	delete[] temp;

	float data[100];
	readConstantF(3, data);
	_press = data[0];
	_tin = data[1];
	_ppm = data[2];

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
	readXSSD(&NISO, &(_x->xddmics(0, 0, 0, 0)));


	readXSDTM(&NISO, &(_x->xdmmicd(0, 0, 0, 0)));
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

	closedb();
	printf("Finished reading burn file : %s\n", dbfile);

}

__host__ void Simon::print(Geometry& g, CrossSection& x, Feedback& f, Depletion& d)
{
	for (int l = 0; l < g.nxyz(); l++)
	{
		printf("%e %e %e %e %e %e %e %e \n", x.xsdf(0, l), x.xstf(0, l), x.chif(0, l), x.xsnf(0, l), x.xssf(0, 1, l), f.tf(l), f.tm(l), d.dnst(2, l));
	}

}