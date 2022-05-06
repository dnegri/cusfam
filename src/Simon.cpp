#include <functional> 
#include "Simon.h"
#include "myblas.h"

extern "C" {

	void readConstantF(const int& n, float* data);
	void writeConstantF(const int& n, float* data);
	void readConstantD(const int& n, double* data);
	void writeConstantD(const int& n, double* data);
	void readConstantI(const int& n, int* data);
	void readString(const int& n, const int& length, char** strings);

	void opendb(int* length, const char* file);
	void closedb();

	//void readDimension(int* ng, int* nxy, int* nz, int* nx, int* ny, int* nsurf);
	//void readIndex(int* nx, int* ny, int* nxy, int* nz, int* nxs, int* nxe, int* nys, int* nye, int* ijtol, int* rotflg, int* neibr,
	//	float* hmesh);

	void* readTableSet(int* length, const char* file, const int* ncomp, char** compnames);
	void* readFormFunction(int* length, const char* file, const int* nhff, char** hffnames);
	void readComposition(const int* nxy, const int*nz, int* ncomp, char** names, int* comps);
	void readDensity(const int& niso, const int& nxyz, float* dnst);
	void writeDensity(const int& niso, const int& nxyz, float* dnst);
	void readStep(float* plevel, float* bucyc, float* buavg, float* efpd, double* eigv, double* fnorm);
	void writeStep(float* plevel, float* bucyc, float* buavg, float* efpd, double* eigv, double* fnorm);

	//void readBoundary(int* symopt, int* symang, float* albedo);
	//void readNXNY(const int* nx, const int* ny, float* val);
	//void readNXYZ(const int* nxyz, float* val);
	//void readNXYZ8(const int* nxyz, double* val);
	//void readNXYZI(const int* nxyz, int* val);

	//void readXS(const int* niso, const int* nxyz, float* xs);
	//void readXSS(const int* niso, const int* nxyz, float* xs);
	//void readXSD(const int* niso, const int* nxyz, float* xs);
	//void readXSSD(const int* niso, const int* nxyz, float* xs);
	//void readXSDTM(const int* niso, const int* nxyz, float* xs);
	//void readXSSDTM(const int* niso, const int* nxyz, float* xs);


	void calculateReference(void* tset_ptr, const int& icomp, const XS_VAR& burn, const XS_VAR* xsmicd, const XS_VAR* xsmica, const XS_VAR* xsmicn,
		const XS_VAR* xsmicf, const XS_VAR* xsmick, const XS_VAR* xsmics, const XS_VAR* xsmic2n, const XS_VAR* xehfp);

	void calculateVariation(void* tset_ptr, const int& icomp, const XS_VAR& burn, const XS_VAR& b10wp,
		const XS_VAR*xdpmicn, const XS_VAR*xdfmicn, const XS_VAR* xdmmicn, const XS_VAR* xddmicn,
		const XS_VAR* xdpmicf, const XS_VAR* xdfmicf, const XS_VAR* xdmmicf, const XS_VAR* xddmicf,
		const XS_VAR* xdpmica, const XS_VAR* xdfmica, const XS_VAR* xdmmica, const XS_VAR* xddmica,
		const XS_VAR* xdpmicd, const XS_VAR* xdfmicd, const XS_VAR* xdmmicd, const XS_VAR* xddmicd,
		const XS_VAR* xdpmics, const XS_VAR* xdfmics, const XS_VAR* xdmmics, const XS_VAR* xddmics);

	void calculateReflector(void* tset_ptr, const int & irefl, const XS_VAR& b10wp,
		const XS_VAR * xsmica, const XS_VAR * xsmicd, const XS_VAR * xsmics,
		const XS_VAR * xdpmica, const XS_VAR * xdmmica, const XS_VAR * xddmica,
		const XS_VAR * xdpmicd, const XS_VAR * xdmmicd, const XS_VAR * xddmicd,
		const XS_VAR * xdpmics, const XS_VAR * xdmmics, const XS_VAR * xddmics);

}

Simon::Simon() {
	_epsbu = 5.0;
	_tset_ptr = NULL;
	_ff_ptr = NULL;
	_fxy = 0.0;
}

Simon::~Simon() {

}

void Simon::initialize(const char* dbfile) {


	int length = strlen(dbfile);
	opendb(&length, dbfile);

	
	int symopt;
	int symang;
	float albedo[6];

	readConstantI(1, &symopt);
	readConstantI(1, &symang);
	readConstantF(6, albedo);

	int ng;
	int nz;
	int nxa;
	int nya;
	int nxya;
	int nxyfa;
	int ncellfa;
	int nxyz;
	int ndivxy;
	int nsubx;
	int nsuby;

	readConstantI(1, &ng);
	readConstantI(1, &nz);
	readConstantI(1, &nxa);
	readConstantI(1, &nya);
	readConstantI(1, &nxya);
	readConstantI(1, &nxyfa);
	readConstantI(1, &ncellfa);
	readConstantI(1, &ndivxy);
	readConstantI(1, &nsubx);
	readConstantI(1, &nsuby);

	int nx;
	int ny;
	int nxy;

	readConstantI(1, &nx);
	readConstantI(1, &ny);
	readConstantI(1, &nxy);
	nxyz = nxy * nz;


	int* nxs = new int[ny];
	int* nxe = new int[ny];
	int* nys = new int[nx];
	int* nye = new int[nx];

	readConstantI(ny, nxs);
	readConstantI(ny, nxe);
	readConstantI(nx, nys);
	readConstantI(nx, nye);

	int nsurf;
	readConstantI(1, &nsurf);
	nsurf = nsurf * nz + (nz + 1) * nxy;

	int ncorn;
	readConstantI(1, &ncorn);

	_g = new Geometry();

	int* ijtol = new int[nx * ny];
	int* neibr = new int[NEWS * nxy];
	float* hmesh = new float[NDIRMAX * nxyz];
	int* latol = new int[NEWS*nxya];
	int* larot = new int[NEWS*nxya];

	readConstantI(nx*ny, ijtol);
	readConstantI(NEWS * nxy, neibr);
	readConstantF(NDIRMAX * nxyz, hmesh);

	readConstantI(NEWS*nxya, latol);
	readConstantI(NEWS*nxya, larot);

	int* lctol = new int[NEWS * ncorn];
	int* ltolc = new int[NEWS * nxy];
	readConstantI(NEWS*ncorn, lctol);
	readConstantI(NEWS*nxy, ltolc);

	_g->setBoundaryCondition(&symopt, &symang, albedo);
	_g->initDimension(&ng, &nxy, &nz, &nx, &ny, &nsurf);
	_g->initIndex(nxs, nxe, nys, nye, ijtol, neibr, hmesh);
	_g->initAssemblyIndex(nxyfa, ncellfa, latol, larot);
	_g->initCorner(ncorn, lctol, ltolc);

	readComposition(&nxy, &nz, &_g->ncomp(), _g->compnames(), _g->comps());
	readComposition(&nxy, &nz, &_g->nhff(), _g->hffnames(), _g->hffs());

	_r = new ControlRod(*_g);

	readConstantI(1, &_r->ncea());
	readString(_r->ncea(), LEN_ROD_NAME, _r->idcea());
	readConstantI(_r->ncea(), _r->abstype());
	readConstantI(nxy, _r->ceamap());

	for (int l = 0; l < nxy; l++) _r->cea(l) = _r->cea(l) - 1;

	_x = new CrossSection(ng, nxy, nxyz);

	_d = new Depletion(*_g);
	_d->init();

	float b10ap;
	readConstantF(1, &b10ap);
	_d->updateB10Abundance(b10ap);

	readConstantF(1, &_pload0);
	_pload = _pload0 * _g->part();


	readConstantF(1, &_d->totmass());
	readConstantF(nxyz, _d->buconf());

	_steam = new SteamTable();
	_steam->setPressure(155.13);

	_f = new Feedback(*_g, *_steam);
	_f->allocate();

	readConstantF(nxy, &(_f->chflow(0)));
	readConstantF(_g->nxyz(), &(_f->ppm0(0)));
	readConstantF(_g->nxyz(), &(_f->stf0(0)));
	readConstantF(_g->nxyz(), &(_f->tm0(0)));
	readConstantF(_g->nxyz(), &(_f->dm0(0)));
	readConstantF(_g->nxyz(), &(_d->h2on(0)));

	readConstantI(nxyz, &(_f->fueltype(0)));
	readConstantF(nxy, &(_f->frodn(0)));
	readConstantI(1, &(_f->nft()));

	_f->initTFTable(_f->nft());
	readConstantI(_f->nft(), &(_f->ntfbu(0)));
	readConstantI(_f->nft(), &(_f->ntfpow(0)));

	int size = TF_POINT * _f->nft();
	readConstantF(size, &(_f->tfbu(0, 0)));
	readConstantF(size, &(_f->tfpow(0, 0)));

	size = size * TF_POINT;
	readConstantF(size, &(_f->tftable(0, 0, 0)));


	_power = new float[_g->nxyz()]{};
	_pow1d = new float[_g->nz()]{};
	_pow2d = new float[_g->nxy()]{};
	_pow2da = new float[_g->nxya()]{};
	_pow3da = new float[_g->nz()*_g->nxya()]{};

	_flux = new SOL_VAR[_g->ngxyz()]{};
	_jnet = new SOL_VAR[_g->nsurf() * _g->ng()]{};
	_phis = new SOL_VAR[_g->nsurf() * _g->ng()]{};

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
	_tset_ptr = ::readTableSet(&length, tsetfile, &_g->ncomp(), _g->compnames());
}

void Simon::readFormFunction(const char * fffile)
{
	int length = strlen(fffile);
	_ff_ptr = ::readFormFunction(&length, fffile, &_g->nhff(), _g->hffnames());
}


void Simon::updateBurnup()
{
	float temp[2*40];
	float temptm[2 * 40 * 3] ;
	float temps[2 *2* 40];
	float tempstm[2 * 2 * 40 * 3];
	float rb10wp = 1. / _d->b10wp();

#pragma omp parallel for
	for (int l = 0; l < _g->nxyz(); l++)
	{

		if (_g->comp(l) > 0) {


			calculateReference(_tset_ptr, _g->comp(l), _d->burn(l),
				_x->xsmicd0(l), _x->xsmica0(l), _x->xsmicn0(l), _x->xsmicf0(l),
				_x->xsmick0(l), _x->xsmics0(l), _x->xsmic2n(l), _x->xehfp(l));

			calculateVariation(_tset_ptr, _g->comp(l), _d->burn(l), rb10wp,
				_x->xdpmicn(l), _x->xdfmicn(l), _x->xdmmicn(l), _x->xddmicn(l),
				_x->xdpmicf(l), _x->xdfmicf(l), _x->xdmmicf(l), _x->xddmicf(l),
				_x->xdpmica(l), _x->xdfmica(l), _x->xdmmica(l), _x->xddmica(l),
				_x->xdpmicd(l), _x->xdfmicd(l), _x->xdmmicd(l), _x->xddmicd(l),
				_x->xdpmics(l), _x->xdfmics(l), _x->xdmmics(l), _x->xddmics(l));

		}
		else {
			calculateReflector(_tset_ptr, _g->comp(l), rb10wp,
				_x->xsmica0(l), _x->xsmicd0(l), _x->xsmics0(l),
				_x->xdpmica(l), _x->xdmmica(l), _x->xddmica(l),
				_x->xdpmicd(l), _x->xdmmicd(l), _x->xddmicd(l),
				_x->xdpmics(l), _x->xdmmics(l), _x->xddmics(l));

		}
	}

	_x->updateMacroXS(&(_d->dnst(0, 0)));
	_x->updateXS(&(_d->dnst(0, 0)), &(_f->dppm(0)), &(_f->dtf(0)), &(_f->dtm(0)));


}

void Simon::setBurnup(const char* dir_burn, const float& burnup) {

	int i = 0;
	for (; i < _nstep; ++i) {
		if (burnup - 10.0 < _bucyc[i]) break;
	}
	
	if (i >= _nstep) {
		i = _nstep - 1;
	}

	char dbfile[512];

	int intbu = round(_bucyc[i]);
	sprintf(dbfile, "%s.%05d.SMR", dir_burn, intbu);
	printf("Started reading burn file : %s\n", dbfile);

	int length = strlen(dbfile);
	opendb(&length, dbfile);
	 
	float bucyc, buavg, efpd;
	readStep(&_pload, &bucyc, &buavg, &efpd, &_eigv, &_fnorm);

	_reigv = 1. / _eigv;


	readDensity(NISO, _g->nxyz(), _d->dnst());

	//if (burnup == 0.0) {
	//	std::copy_n(_d->dnst_new(), NISO * _g->nxyz(), _d->dnst());
	//}


	readConstantF(_g->nxyz(), _d->burn());
	readConstantF(_g->nxyz(), _power);

	double* temp = new double[_g->ngxyz()];
	readConstantD(_g->ngxyz(), temp);
	std::copy_n(temp, _g->ngxyz(), _flux);
	delete[] temp;

	//float data[100];
	//readConstantF(3, data);
	//_press = data[0];::
	//_tin = data[1];
	//_ppm = data[2];

	//readNXYZ(&(_g->nxyz()), &(_f->tf(0)));
	//readNXYZ(&(_g->nxyz()), &(_f->tm(0)));
	//readNXYZ(&(_g->nxyz()), &(_f->dm(0)));


	//_f->updatePressure(_press);
	//_f->updateTin(_tin);
	//_f->initDelta(_ppm);
	//_d->updateH2ODensity(_f->dm(), _ppm);


	//readXS(&NISO, &(_g->nxyz()), &(_x->xsmicd0(0, 0, 0)));
	//readXS(&NISO, &(_g->nxyz()), &(_x->xsmica0(0, 0, 0)));
	//readXS(&NISO, &(_g->nxyz()), &(_x->xsmicf0(0, 0, 0)));
	//readXS(&NISO, &(_g->nxyz()), &(_x->xsmicn0(0, 0, 0)));
	//readXS(&NISO, &(_g->nxyz()), &(_x->xsmick0(0, 0, 0)));
	//readXSS(&NISO,&(_g->nxyz()),  &(_x->xsmics0(0, 0, 0, 0)));

	//readXSD(&NISO, &(_g->nxyz()), &(_x->xdpmicd(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xdpmica(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xdpmicf(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xdpmicn(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xdpmick(0, 0, 0)));
	//readXSSD(&NISO,&(_g->nxyz()),  &(_x->xdpmics(0, 0, 0, 0)));

	//readXSD(&NISO, &(_g->nxyz()), &(_x->xdfmicd(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xdfmica(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xdfmicf(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xdfmicn(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xdfmick(0, 0, 0)));
	//readXSSD(&NISO,&(_g->nxyz()),  &(_x->xdfmics(0, 0, 0, 0)));

	//readXSD(&NISO, &(_g->nxyz()), &(_x->xddmicd(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xddmica(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xddmicf(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xddmicn(0, 0, 0)));
	//readXSD(&NISO, &(_g->nxyz()), &(_x->xddmick(0, 0, 0)));
	//readXSSD(&NISO,&(_g->nxyz()),  &(_x->xddmics(0, 0, 0, 0)));


	//readXSDTM(&NISO, &(_g->nxyz()), &(_x->xdmmicd(0, 0, 0, 0)));
	//readXSDTM(&NISO, &(_g->nxyz()), &(_x->xdmmica(0, 0, 0, 0)));
	//readXSDTM(&NISO, &(_g->nxyz()), &(_x->xdmmicf(0, 0, 0, 0)));
	//readXSDTM(&NISO, &(_g->nxyz()), &(_x->xdmmicn(0, 0, 0, 0)));
	//readXSDTM(&NISO, &(_g->nxyz()), &(_x->xdmmick(0, 0, 0, 0)));
	//readXSSDTM(&NISO,&(_g->nxyz()),  &(_x->xdmmics(0, 0, 0, 0, 0)));

	//_x->updateMacroXS(&(_d->dnst(0, 0)));
	//_x->updateXS(&(_d->dnst(0, 0)), &(_f->dppm(0)), &(_f->dtf(0)), &(_f->dtm(0)));

	closedb();
	printf("Finished reading burn file : %s\n", dbfile);

	_f->updatePressure(155.13);

}


void Simon::setSMR(const char* dir_burn, const float& burnup) {

	char dbfile[512];

	int intbu = round(burnup);
	sprintf(dbfile, "%s.%05d.SSMR", dir_burn, intbu);
	printf("Started reading burn file : %s\n", dbfile);

	int length = strlen(dbfile);
	opendb(&length, dbfile);

	float bucyc, buavg, efpd;
	readStep(&_pload, &bucyc, &buavg, &efpd, &_eigv, &_fnorm);

	_reigv = 1. / _eigv;


	readDensity(NISO, _g->nxyz(), _d->dnst());

	readConstantF(_g->nxyz(), _d->burn());
	readConstantF(_g->nxyz(), _power);

	double* temp = new double[_g->ngxyz()];
	readConstantD(_g->ngxyz(), temp);
	std::copy_n(temp, _g->ngxyz(), _flux);
	delete[] temp;

	closedb();
	printf("Finished reading burn file : %s\n", dbfile);

	_f->updatePressure(155.13);

}

void Simon::saveSMR(const char* dir_burn, const float& burnup) {

	int i = 0;
	for (; i < _nstep; ++i) {
		if (burnup - 10.0 < _bucyc[i]) break;
	}

	char dbfile[512];

	int intbu = round(_bucyc[i]);
	sprintf(dbfile, "%s.%05d.SSMR", dir_burn, intbu);
	printf("Started saving burn file : %s\n", dbfile);

	int length = strlen(dbfile);
	opendb(&length, dbfile);

	float bucyc, buavg, efpd;
	writeStep(&_pload, &bucyc, &buavg, &efpd, &_eigv, &_fnorm);
	   
	writeDensity(NISO, _g->nxyz(), _d->dnst());
	
	writeConstantF(_g->nxyz(), _d->burn());
	writeConstantF(_g->nxyz(), _power);

	double* temp = new double[_g->ngxyz()];
	writeConstantD(_g->ngxyz(), temp);
	std::copy_n(temp, _g->ngxyz(), _flux);
	delete[] temp;

	closedb();
	printf("Finished saving burn file : %s\n", dbfile);

}

__host__ void Simon::setBurnupPoints(const std::vector<double>& burnups)
{
	//readBurnupList
	_nstep = burnups.size();
	_bucyc = new float[_nstep]; 

	int i = -1;
	for (const double & b : burnups) {
		_bucyc[++i] = b;
	}
}


__host__ void Simon::setRodPosition(const char* rodid, const float& position)
{
	_r->setPosition(rodid, position);
}

__host__ void Simon::print(Geometry& g, CrossSection& x, Feedback& f, Depletion& d)
{
	for (int l = 0; l < g.nxyz(); l++)
	{
		printf("%e %e %e %e %e %e %e %e \n", x.xsdf(0, l), x.xstf(0, l), x.chif(0, l), x.xsnf(0, l), x.xssf(0, 1, l), f.tf(l), f.tm(l), d.dnst(2, l));
	}

}

void Simon::generateResults()
{
	int l = -1;

	std::fill(pow1d(), pow1d() + _g->nz(), 0.0);
	std::fill(pow2d(), pow2d() + _g->nxy(), 0.0);
	for (int k = 0; k < _g->nz(); k++) {
		for (int l2d = 0; l2d < _g->nxy(); l2d++)
		{
			++l;
			pow1d(k) += power(l);
			pow2d(l2d) += power(l);
		}
	}

	float pow1d_up = 0.0;
	float pow1d_low = 0.0;

	float hz_half = _g->hzcore() * 0.5;
	float hz = 0.0;
	for (int k = _g->kbc(); k < _g->kec(); k++)
	{
		hz += _g->hmesh(ZDIR, k* _g->nxy());

		if (hz >= hz_half + EPS_ROD_IN) {
			pow1d_up += pow1d(k);
		}
		else {
			pow1d_low += pow1d(k);
		}
	}

	_asi = (pow1d_low - pow1d_up) / (pow1d_low + pow1d_up);

	std::fill(pow2da(), pow2da() + _g->nxya(), 0.0);

	for (int l2d = 0; l2d < _g->nxy(); l2d++)
	{
		int l2da = _g->ltola(l2d);
		pow2da(l2da) += pow2d(l2d);
	}

	//FIXME volcore should be given by Geometry.
	GEOM_VAR volcore = 0.0;
	GEOM_VAR hzcore = _g->hzcore();

	for (int l = 0; l < _g->nxyz(); l++)
	{
		if(_x->xsmacn0(1, l) != 0.0)	volcore += _g->vol(l);
	}

	
	for (int l2da = 0; l2da < _g->nxya(); l2da++)
	{
		pow2da(l2da) = pow2da(l2da) / _pload / _g->vola(l2da) * volcore / hzcore * _g->hmesh(ZDIR, 0);
	}
	std::fill(pow3da(), pow3da() + _g->nxya()* _g->nz(), 0.0);
	for (int k = _g->kbc(); k < _g->kec(); k++)
	{
		for (int l2d = 0; l2d < _g->nxy(); l2d++)
		{
			auto l2da = _g->ltola(l2d);
			pow3da(l2da, k) += power(l2d, k);
		}

		for (int l2da = 0; l2da < _g->nxya(); l2da++)
		{
			auto lka = k * _g->nxya() + l2da;
			pow3da(l2da, k) = pow3da(l2da, k) / _g->vola(lka) / _pload * volcore;
		}

	}



	l = 0;
	for (int k = 0; k < _g->nz(); k++) {
		pow1d(k) = pow1d(k) / _pload / _g->hmesh(ZDIR, l) * hzcore;
		l += _g->nxy();
	}

}