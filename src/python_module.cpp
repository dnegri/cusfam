#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Python.h>

#include <Windows.h>
#include <cmath>
#include <fenv.h>
#include "omp.h"
#include "SimonCPU.h"
#include "plog/Log.h"
#include "plog/Appenders/ConsoleAppender.h"

namespace py = pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<float>);

template <class T> class ptr_wrapper
{
public:
	ptr_wrapper() : ptr(nullptr) {}
	ptr_wrapper(T* ptr) : ptr(ptr) {}
	ptr_wrapper(const ptr_wrapper& other) : ptr(other.ptr) {}
	T& operator* () const { return *ptr; }
	T* operator->() const { return  ptr; }
	T* get() const { return ptr; }
	void destroy() { delete ptr; }
	T& operator[](std::size_t idx) const { return ptr[idx]; }
private:
	T* ptr;
};

static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;

unique_ptr<SimonCPU> simon_init(const char* geom_file, const char* tset_file, const char* ff_file) {
	//omp_set_num_threads(4);

	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	//feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	
	plog::init(plog::warning, &consoleAppender);

	auto simon = unique_ptr<SimonCPU>(new SimonCPU());
	simon->initialize(geom_file);
	simon->readTableSet(tset_file);
	simon->readFormFunction(ff_file);

	return simon;
}

void simon_getGeometry(SimonCPU& simon, SimonGeometry& sg) { //, py::array_t<float> hz) {
	Geometry& g = simon.g();
	

	sg.nxya = g.nxya();
	sg.nz = g.nz();
	sg.kbc = g.kbc();
	sg.kec = g.kec();

	sg.core_height = g.hzcore();

	for (int k=0; k < g.nz(); k++) 
	{
		sg.hz.append(g.hmesh(ZDIR, k* g.nxy()));
	}
}

void simon_setBurnupPoints(SimonCPU& simon, std::vector<double> burnups) {

	simon.setBurnupPoints(burnups);
}

void simon_setBurnup(SimonCPU & simon, const char* dir_burnup, float burnup) {
	simon.setBurnup(dir_burnup, burnup);
	simon.updateBurnup();
}

void simon_calcStatic(SimonCPU& simon, SteadyOption& s) {
	printf("begin\n");
	simon.runSteady(s);
	printf("end\n");
	//simon.runSteadySfam(s);
}

void simon_calcPinPower(SimonCPU& simon) {
	simon.runPinPower();
	//simon.runSteadySfam(s);
}


void simon_getResult(SimonCPU& simon, SimonResult& result) {
	simon.generateResults();
	result.ppm = simon.ppm();
	result.eigv = simon.eigv();
	result.asi = simon.asi();
	result.fxy = simon.fxy();
	result.fr  = simon.fr();
	result.fq  = simon.fq();
	Geometry& g = simon.g();

	for (int la = 0; la < g.nxya(); la++)
	{
		result.pow2d[la] = simon.pow2da(la);
	}

	for (int k = 0; k < g.nz(); k++)
	{
		result.pow1d[k] = simon.pow1d(k);
	}
}

void simon_setRodPosition(SimonCPU& simon, const char* rodid, const float& position) {
	simon.setRodPosition(rodid, position);
}

void simon_deplete(SimonCPU& simon, XEType xe_option, SMType sm_option, float del_burnup) {

	DepletionOption option;
	option.tsec = del_burnup / simon.pload() * simon.d().totmass() * 3600.0 * 24.0;
	option.xe = xe_option;
	option.sm = sm_option;

	simon.runDepletion(option);
	simon.updateBurnup();
}

void simon_depleteByTime(SimonCPU& simon, XEType xe_option, SMType sm_option, float tsec) {

	DepletionOption option;
	option.tsec = tsec;
	option.xe = xe_option;
	option.sm = sm_option;

	simon.runDepletion(option);
	simon.updateBurnup();
}

void simon_depleteXeSm(SimonCPU& simon, XEType xe_option, SMType sm_option, float tsec) {

	DepletionOption option;
	option.tsec = tsec;
	option.xe = xe_option;
	option.sm = sm_option;

	simon.runXenonTransient(option);
}

namespace py = pybind11;

PYBIND11_MODULE(cusfam, m) {
	py::class_<SimonCPU, unique_ptr<SimonCPU>>(m, "SimonCPU");

	py::class_<ptr_wrapper<float>>(m, "pfloat");

	py::class_<SimonGeometry>(m, "SimonGeometry")
		.def(py::init())
		.def_readwrite("nz", &SimonGeometry::nz)
		.def_readwrite("nxya", &SimonGeometry::nxya)
		.def_readwrite("kbc", &SimonGeometry::kbc)
		.def_readwrite("kec", &SimonGeometry::kec)
		.def_readwrite("core_height", &SimonGeometry::core_height)
		.def_readwrite("hz", &SimonGeometry::hz);

	py::class_<SimonResult>(m, "SimonResult")
		.def(py::init<int, int>())
		.def_readwrite("error", &SimonResult::error)
		.def_readwrite("eigv", &SimonResult::eigv)
		.def_readwrite("ppm", &SimonResult::ppm)
		.def_readwrite("fq", &SimonResult::fq)
		.def_readwrite("fxy", &SimonResult::fxy)
		.def_readwrite("fr", &SimonResult::fr)
		.def_readwrite("fz", &SimonResult::fz)
		.def_readwrite("asi", &SimonResult::asi)
		.def_readwrite("tf", &SimonResult::tf)
		.def_readwrite("tm", &SimonResult::tm)
		.def_readwrite("rod_pos", &SimonResult::rod_pos)
		.def_readwrite("pow2d", &SimonResult::pow2d)
		.def_readwrite("pow1d", &SimonResult::pow1d);

	py::class_<SteadyOption>(m, "SteadyOption")
		.def(py::init())
		.def_readwrite("crit", &SteadyOption::searchOption)
		.def_readwrite("feedtf", &SteadyOption::feedtf)
		.def_readwrite("feedtm", &SteadyOption::feedtm)
		.def_readwrite("xenon", &SteadyOption::xenon)
		.def_readwrite("tin", &SteadyOption::tin)
		.def_readwrite("eigvt", &SteadyOption::eigvt)
		.def_readwrite("maxiter", &SteadyOption::maxiter)
		.def_readwrite("epsiter", &SteadyOption::epsiter)
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
        Initialize SimonCPU instance.
    )pbdoc");

	m.def("getGeometry", &simon_getGeometry, R"pbdoc(
        Initialize SimonCPU instance.
    )pbdoc");

	m.def("setBurnupPoints", &simon_setBurnupPoints, R"pbdoc(
        Set burnup points.
    )pbdoc");

	m.def("setBurnup", &simon_setBurnup, R"pbdoc(
        Set a core burnup point.
    )pbdoc");

	m.def("setRodPosition", &simon_setRodPosition, R"pbdoc(
        Set control rod position with rod id.
    )pbdoc");

	m.def("calcStatic", &simon_calcStatic, R"pbdoc(
        Run steady-state calculation.
    )pbdoc");

	m.def("calcPinPower", &simon_calcPinPower, R"pbdoc(
        Run steady-state calculation.
    )pbdoc");

	m.def("deplete", &simon_deplete, R"pbdoc(
        Run Depletion calculation.
    )pbdoc");

	m.def("depleteByTime", &simon_depleteByTime, R"pbdoc(
        Run Depletion calculation.
    )pbdoc");

	m.def("depleteXeSm", &simon_depleteXeSm, R"pbdoc(
        Run Xenon & Samarium Dynamics.
    )pbdoc");

	m.def("getResult", &simon_getResult, R"pbdoc(
        get result.
    )pbdoc");


#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}