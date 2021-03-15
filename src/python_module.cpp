#include <pybind11/pybind11.h>
#include <Python.h>

#include <Windows.h>
#include <cmath>
#include <fenv.h>
#include "omp.h"
#include "SimonCPU.h"
#include "plog/Log.h"
#include "plog/Appenders/ConsoleAppender.h"

extern "C" {
	void initsfamgeom(const int& ng, const int& ndim, const int& ngeo,
		const int& nx, const int& ny, const int& nz, const int& nxy,
		const int& kbc, const int& kec, const int* nxs, const int* nxe, const int* nys, const int* nye,
		const int* ijtol, const GEOM_VAR* hmesh, const GEOM_VAR* vol, const GEOM_VAR* albedo);

	void initsfamxsec(const int& ng, const int& ng2s, const int& ndim, const int& nxy, const int& nz,
		const XS_VAR* xschif, const XS_VAR* xsdf, const XS_VAR* xstf, const XS_VAR* xsnf,
		const XS_VAR* xskf, const XS_VAR* xssf);
	
}

SimonCPU* simon;
char dir_burnup[MAX_PATH];

static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;

void simon_init(const char* geom_file, const char* tset_file, const char* dir_burnup_) {
	omp_set_num_threads(4);
	//feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	plog::init(plog::warning, &consoleAppender);
	simon = new SimonCPU();
	simon->initialize(geom_file);
	simon->readTableSet(tset_file);
	strcpy_s(dir_burnup, dir_burnup_);

	Geometry& g = simon->g();
	CrossSection& x = simon->x();

	initsfamgeom(g.ng(), 3, 4, g.nx(), g.ny(), g.nz(), g.nxy(),
		2, 25, g.nxs(), g.nxe(), g.nys(), g.nye(), g.ijtol(), g.hmesh(), g.vol(), g.albedo());

	initsfamxsec(g.ng(), 1, 3, g.nxy(), g.nz(), x.chif(), x.xsdf(), x.xstf(), x.xsnf(), x.xskf(), x.xssf());
}

void simon_setBurnup(float burnup) {
	simon->setBurnup(dir_burnup, burnup);
	simon->updateBurnup();
}

void simon_calcStatic(SteadyOption& s) {
	simon->runSteadySfam(s);
	//simon->runSteady(s);
	s.ppm = simon->ppm();

}

void simon_deplete(XEType xe_option, SMType sm_option, float del_burnup) {

	DepletionOption option;
	option.tsec = del_burnup / simon->pload() * simon->d().totmass() * 3600.0 * 24.0;
	option.xe = xe_option;
	option.sm = sm_option;

	simon->runDepletion(option);
	simon->updateBurnup();
}


void simon_shutdownmargin(SteadyOption& s) {
	simon->runSteady(s);
}


namespace py = pybind11;

PYBIND11_MODULE(cusfam, m) {
	py::class_<SteadyOption>(m, "SteadyOption")
		.def(py::init())
		.def_readwrite("crit", &SteadyOption::searchOption)
		.def_readwrite("feedtf", &SteadyOption::feedtf)
		.def_readwrite("feedtm", &SteadyOption::feedtm)
		.def_readwrite("xenon", &SteadyOption::xenon)
		.def_readwrite("tin", &SteadyOption::tin)
		.def_readwrite("eigvt", &SteadyOption::eigvt)
		.def_readwrite("maxiter", &SteadyOption::maxiter)
		.def_readwrite("ppm", &SteadyOption::ppm)
		.def_readwrite("plevel", &SteadyOption::plevel);

	py::enum_<CriticalOption>(m, "CriticalOption")
		.value("KEFF", CriticalOption::KEFF)
		.value("CBC", CriticalOption::CBC)
		.export_values();

	py::enum_<XEType>(m, "XEType")
		.value("XE_NO", XEType::XE_NO)
		.value("XE_EQ", XEType::XE_EQ)
		.value("XE_TR", XEType::XE_TR)
		.export_values();

	py::enum_<SMType>(m, "SMType")
		.value("SM_NO", SMType::SM_NO)
		.value("SM_TR", SMType::SM_TR)
		.export_values();

	py::enum_<DepletionIsotope>(m, "DepletionIsotope")
		.value("DEP_ALL", DepletionIsotope::DEP_ALL)
		.value("DEP_FP", DepletionIsotope::DEP_FP)
		.value("DEP_XE", DepletionIsotope::DEP_XE)
		.export_values();

	m.def("init", &simon_init, R"pbdoc(
        Initialize Simon instance.
    )pbdoc");

	m.def("setBurnup", &simon_setBurnup, R"pbdoc(
        Set a core burnup point.
    )pbdoc");

	m.def("calcStatic", &simon_calcStatic, R"pbdoc(
        Run steady-state calculation.
    )pbdoc");

	m.def("deplete", &simon_deplete, R"pbdoc(
        Run Depletion calculation.
    )pbdoc");

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}