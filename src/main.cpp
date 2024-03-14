#include "ASISearcher.h"
#include "BICGCMFD.h"
#include "CMFDCPU.h"
#include "CrossSection.h"
#include "Depletion.h"
#include "Feedback.h"
#include "Geometry.h"
#include "NodalCPU.h"
#include "SimonCPU.h"
#include "Snapshot.h"
#include "omp.h"
#include "pch.h"
#include <time.h>
#include "plog/Appenders/ConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Init.h"
#include "plog/Log.h"

#include <fenv.h>

#ifndef CPU
    #include "CrossSectionCuda.h"
    #include "GeometryCuda.h"
    #include "SimonCuda.h"
#endif

#ifndef CPU
dim3 BLOCKS_NGXYZ;
dim3 THREADS_NGXYZ;
dim3 BLOCKS_NODE;
dim3 THREADS_NODE;
dim3 BLOCKS_2D;
dim3 THREADS_2D;
dim3 BLOCKS_SURFACE;
dim3 THREADS_SURFACE;
#endif

static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;

int64_t dnegri::jiarray::sizeOfJIArray = 0;

int main() {
    // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    // #ifdef _OPENMP
    omp_set_num_threads(8);

    // #endif

    plog::init(plog::debug, &consoleAppender);

    SimonCPU     simon;
    SteadyOption s;

    // simon.initialize("C:/simon/restart_files/UCN613ASBDEP.SMG");
    // simon.readTableSet("C:/simon/plant_files/OPR1000_rev1.XS");
    // simon.readFormFunction("C:/simon/plant_files/OPR1000_rev1.FF");
    // simon.setBurnupPoints({0.0, 50.0, 150.0, 500.0, 1000.0,2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0});
    // simon.setBurnup("C:/simon/restart_files/UCN613ASBDEP", 0.0, s);

    // vector<double> burnupPoints = {0.0, 50.0, 150.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0,
    //                                9000.0, 10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000, 18000, 18834};
    // simon.initialize("./run/skn3/S304NDR.SMG");
    // simon.readTableSet("./run/skn3/PLUS7_V127.XS");
    // simon.readFormFunction("./run/skn3/PLUS7_V127.FF");
    // simon.setBurnupPoints(burnupPoints);
    // simon.setBurnup("./run/skn3/S304NDR", 15000.0, s);

    // vector<double> burnupPoints = {0.0, 50.0, 150.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0,
    //                                9000.0, 10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000, 18000, 18834};
    // simon.initialize("./run/ucn6/UCN612ASBDEP.SMG");
    // simon.readTableSet("./run/ucn6/OPR1000_rev1.XS");
    // simon.readFormFunction("./run/ucn6/OPR1000_rev1.FF");
    // simon.setBurnupPoints(burnupPoints);
    // simon.setBurnup("./run/ucn6/UCN612ASBDEP", 0.0, s);

    vector<double> burnupPoints = {0.0, 50.0, 150.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0,
                                   9000.0, 10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000};
    simon.initialize("./run/ygn3/Y312ASBDEP.SMG");
    simon.readTableSet("./run/ygn3/KMYGN34C01_PLUS7_XSE.XS");
    simon.readFormFunction("./run/ygn3/KMYGN34C01_PLUS7_XSE.FF");
    simon.setBurnupPoints(burnupPoints);

    s.plevel       = 1.0;
    s.ppm          = 1000.0;
    s.tin          = 290.0;
    s.shpmtch      = ShapeMatchOption::SHAPE_NO;
    s.searchOption = CriticalOption::CBC;
    s.xenon        = XEType::XE_EQ;
    s.samarium     = SMType::SM_TR;
    s.feedtm       = true;
    s.feedtf       = true;
    s.eigvt        = 1.00070;
    s.epsiter      = 1.E-5;
    s.maxiter      = 100;

    DepletionOption d;
    d.isotope = DepletionIsotope::DEP_ALL;
    d.sm      = SMType::SM_TR;
    d.xe      = XEType::XE_EQ;
    d.tsec    = 30*24*3600.0;

    simon.setBurnup("./run/ygn3/Y312ASBDEP", burnupPoints[10], s);

    for (size_t i = 0; i < 1; i++) {
        // simon.setBurnup("./run/skn3/S304NDR", burnupPoints[i], s);
        // simon.setBurnup("./run/ucn6/UCN612ASBDEP", burnupPoints[i], s);
        simon.setBurnup("./run/ygn3/Y312ASBDEP", burnupPoints[i], s);

        simon.updateBurnup();
        simon.runSteady(s);
        simon.generateResults();
        // simon.runDepletion(d);
        // d.tsec = d.tsec+1;
        // s.ppm = simon.ppm();
        /* code */
        printf("DEPLETION : %.1f,  ASI : %.3f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f\n", burnupPoints[i], simon.asi(), simon.ppm(), simon.eigv());
    }

    // PLOG_DEBUG << omp_get_thread_num() << "\n";
    exit(0);

    // for (int i = simon.g().nz()-1; i >= 0; --i) {
    //     printf("NODE : %d, POWER : %.4f\n", i+1, simon.pow1d(i));
    // }

    // s.xenon = XEType::XE_TR;
    // s.plevel = 0.0;
    // simon.runSteady(s);
    // simon.generateResults();
    // s.ppm = simon.ppm();

    // printf("DEPLETION : %d,  ASI : %.3f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f\n", 1, simon.asi(), simon.ppm(), simon.eigv());

    // simon.setRodPosition("P", 0.0);
    // simon.runSteady(s);
    // simon.generateResults();
    // printf("DEPLETION : %d,  ASI : %.3f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f\n", 1, simon.asi(), simon.ppm(), simon.eigv());

    // simon.setRodPosition("R", 0.0);
    // simon.runSteady(s);
    // simon.generateResults();
    // printf("DEPLETION : %d,  ASI : %.3f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f\n", 1, simon.asi(), simon.ppm(), simon.eigv());

    // simon.setRodPosition("B", 0.0);
    // simon.setRodPosition("A", 0.0);
    // simon.runSteady(s);
    // simon.generateResults();
    // s.ppm = simon.ppm();

    // printf("DEPLETION : %d,  ASI : %.3f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f\n", 1, simon.asi(), simon.ppm(), simon.eigv());

    // exit(0);

    // s.xenon = XEType::XE_TR;
    // s.plevel = 0.0;

    // simon.setRodPosition("R5", 0.0);
    // simon.setRodPosition("R4", 0.0);
    // simon.setRodPosition("R3", 0.0);
    // simon.setRodPosition("P", 0.0);
    // simon.runSteady(s);
    // simon.generateResults();
    // s.ppm = simon.ppm();

    // printf("DEPLETION : %d,  ASI : %.3f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f/n", 1, simon.asi(), simon.ppm(), simon.eigv());

    // DepletionOption depOption;
    // depOption.tsec = 3600.0;
    // depOption.xe = XE_TR;
    // depOption.sm = SM_TR;
    // depOption.isotope = DepletionIsotope::DEP_ALL;

    // simon.runDepletion(depOption);
    // simon.updateBurnup();
    // simon.runSteady(s);
    // simon.generateResults();
    // s.ppm = simon.ppm();

    // printf("DEPLETION : %d,  ASI : %.3f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f/n", 1, simon.asi(), simon.ppm(), simon.eigv());

    //    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //
    //    for(int i=0; i<simon.nburn();i++) {
    //        simon.setBurnup("./run/ygn3/Y312ASBDEP", simon.burn(i));
    //        simon.updateBurnup();
    //        simon.runSteady(s);
    //        simon.generateResults();
    //        s.ppm = simon.ppm();
    //        printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f/n", 1, 0.0, simon.ppm(), simon.eigv());
    //    }
    //
    //    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //
    //    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    //
    //    exit(0);

    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // s.ppm = simon.ppm();
    // s.xenon = XE_TR;
    // s.searchOption = CriticalOption::CBC;
    // s.feedtm = true;
    // s.feedtf = true;
    // // s.shpmtch = ShapeMatchOption::SHAPE_MATCH;

    // ASISearcher asis(simon.g());
    // asis.search(1.E-4, 10, simon, s, -0.03, 0.001);
    // // asis.search(1.E-4, 10, simon, option, targetASI, 0.001);
    // simon.generateResults();
    // printf("DEPLETION : %d,  ASI : %.3f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f/n", 1, simon.asi(), simon.ppm(), simon.eigv());

    // simon.runSteady(s);
    // simon.generateResults();
    // printf("DEPLETION : %d,  ASI : %.3f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f/n", 1, simon.asi(), simon.ppm(), simon.eigv());

    // for (int i = simon.g().nz()-1; i >= 0; --i) {
    //     printf("NODE : %d, POWER : %.4f\n", i+1, simon.pow1d(i));
    // }

    //    vector<double> p1d;
    //    for (int i = 0; i < simon.g().nz(); ++i) {
    //        p1d.push_back(simon.pow1d(i));
    //    }

    // vector<double> p1d, hz;

    // hz.push_back(9.050);
    // hz.push_back(6.190);
    // hz.push_back(2.860);
    // hz.push_back(10.000);
    // hz.push_back(10.000);
    // hz.push_back(20.000);
    // hz.push_back(18.100);
    // hz.push_back(18.100);
    // hz.push_back(20.000);
    // hz.push_back(20.000);
    // hz.push_back(18.100);
    // hz.push_back(18.100);
    // hz.push_back(20.000);
    // hz.push_back(20.000);
    // hz.push_back(18.100);
    // hz.push_back(18.100);
    // hz.push_back(20.000);
    // hz.push_back(20.000);
    // hz.push_back(18.100);
    // hz.push_back(18.100);
    // hz.push_back(20.000);
    // hz.push_back(10.000);
    // hz.push_back(10.000);
    // hz.push_back(2.860);
    // hz.push_back(6.190);
    // hz.push_back(9.050);

    // p1d.push_back(0.706660000000000 ); //(1)
    // p1d.push_back(0.664816000000000 ); //(2)
    // p1d.push_back(0.398673000000000 ); //(3)
    // p1d.push_back(1.57690300000000  ); //(4)
    // p1d.push_back(1.76141700000000  ); //(5)
    // p1d.push_back(3.68713900000000  ); //(6)
    // p1d.push_back(3.30693200000000  ); //(7)
    // p1d.push_back(3.24469800000000  ); //(8)
    // p1d.push_back(3.58477700000000  ); //(9)
    // p1d.push_back(3.67191600000000  ); //(10)
    // p1d.push_back(3.45467700000000  ); //(11)
    // p1d.push_back(3.60384800000000  ); //(12)
    // p1d.push_back(4.15695500000000  ); //(13)
    // p1d.push_back(4.40367500000000  ); //(14)
    // p1d.push_back(4.34542500000000  ); //(15)
    // p1d.push_back(4.82476600000000  ); //(16)
    // p1d.push_back(6.05931800000000  ); //(17)
    // p1d.push_back(7.02624700000000  ); //(18)
    // p1d.push_back(7.35116499999999  ); //(19)
    // p1d.push_back(8.32742500000000  ); //(20)
    // p1d.push_back(9.91233600000000  ); //(21)
    // p1d.push_back(4.81863500000000  ); //(22)
    // p1d.push_back(4.28267699999999  ); //(23)
    // p1d.push_back(1.07846800000000  ); //(24)
    // p1d.push_back(1.80874700000000  ); //(25)
    // p1d.push_back(1.94254300000000  ); //(26)

    // for (int iter = 0; iter < 10; ++iter) {
    //     simon.setPowerShape(hz, p1d);
    //     simon.runShapeMatch(s);
    //     simon.generateResults();
    //     printf("DEPLETION : %d,  ASI : %.3f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.5f/n", 1, simon.asi(), simon.ppm(), simon.eigv());
    //     exit(-1);
    // }

    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    // exit(0);

    simon.setBurnup("./run/ygn3/Y312ASBDEP", burnupPoints[10], s);
    simon.updateBurnup();
    simon.runSteady(s);

    s.plevel       = 0.9;
    s.ppm          = simon.ppm();
    s.xenon        = XE_TR;
    s.searchOption = CriticalOption::KEFF;

    DepletionOption depletionOption;
    depletionOption.xe   = XEType::XE_TR;
    depletionOption.sm   = SMType::SM_TR;
    depletionOption.tsec = 15 * 60;
    double time = 0.0;
    for (int i = 0; i < 1 * 1344; i++) {
        time += depletionOption.tsec;
        simon.runXenonTransient(depletionOption);
        simon.runSteady(s);
        simon.generateResults();
        PLOG_DEBUG << fmt::format("XENON TRANSIENT - TIME: {:>9.1f} (min), ASI : {:>7.3f}", time/60, simon.asi());
    }
    exit(0);


    Snapshot snapshot(simon.g());
    snapshot.save(simon);

    map<int, Snapshot*> maps;
    maps.insert({1, &snapshot});
    auto a = maps[1];
    for (int i = 0; i < 1 * 200; i++) {
        simon.runXenonTransient(depletionOption);
        simon.runSteady(s);
        simon.generateResults();
        cout << "REF ASI : " << simon.asi() << "\n";
    }
    snapshot.load(simon);
    simon.runSteady(s);
    for (int i = 0; i < 1 * 200; i++) {
        simon.runXenonTransient(depletionOption);
        simon.runSteady(s);
        simon.generateResults();
        cout << "LOAD ASI : " << simon.asi() << "\n";
    }

    exit(0);

    auto eigvd = simon.eigv();

    s.ppm = simon.ppm();

    s.searchOption = CriticalOption::KEFF;
    s.xenon        = XEType::XE_TR;
    s.feedtm       = true;
    s.feedtf       = false;

    //    simon.r().setPosition("R5", 274.3);
    //    simon.runSteady(s);
    //    simon.generateResults();
    //    auto rho = 1. - 1/simon.eigv();
    //    printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.6f/n", 1, 0.0, simon.ppm(), simon.eigv());
    //    printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), RHO : %.1f/n", 1, 0.0, simon.ppm(), rho*1.E+5);
    //
    //    eigvd = simon.eigv();
    //
    s.searchOption = CriticalOption::KEFF;
    s.plevel       = 0.0;
    simon.runSteady(s);
    simon.generateResults();
    printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), RHO : %.1f/n", 1, 0.0, simon.ppm(), (1. - 1 / simon.eigv()) * 1.E+5);

    exit(0);
    //	DepletionOption d_option;
    //	d_option.isotope = DepletionIsotope::DEP_ALL;
    //	d_option.sm = SMType::SM_TR;
    //	d_option.xe = s.xenon;
    //
    //
    //	auto start = chrono::steady_clock::now();
    //    simon.runSteady(s);
    //
    //    s.feedtm = false;
    //    s.feedtf = false;
    //    s.maxiter = 100;
    //    s.xenon = XEType::XE_TR;

    // simon.r().setPosition("R", 0.0);
    // simon.r().setPosition("P", 0.0);
    // simon.r().setPosition("A", 0.0);
    //	simon.r().setPosition(35, 0.0);
    //    simon.runSteady(s);
    //    printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.6f/n", 1, 0.0, simon.ppm(), simon.eigv());
    //
    //    simon.r().setPosition(35, 381.0);
    //    simon.r().setPosition(36, 0.0);
    //	simon.runSteady(s);
    //	printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.6f/n", 1, 0.0, simon.ppm(), simon.eigv());
    //
    //    simon.r().setPosition(36, 381.0);
    //    simon.r().setPosition(39, 0.0);
    //    simon.runSteady(s);
    //    printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.6f/n", 1, 0.0, simon.ppm(), simon.eigv());
    //
    //    simon.r().setPosition(39, 381.0);
    //    simon.r().setPosition(40, 0.0);
    //    simon.runSteady(s);
    //    printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.6f/n", 1, 0.0, simon.ppm(), simon.eigv());
    //
    //    simon.r().setPosition(40, 381.0);
    simon.r().setPosition("R1", 0.0);
    simon.runSteady(s);
    simon.generateResults();
    printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.6f/n", 1, 0.0, simon.ppm(), simon.eigv());

    simon.r().setPosition("B22", 381.0);
    simon.r().setPosition("B42", 0.0);
    simon.runSteady(s);
    simon.generateResults();
    printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.6f/n", 1, 0.0, simon.ppm(), simon.eigv());

    exit(0);

    //	s.ppm = simon.ppm();
    //	s.searchOption = CriticalOption::KEFF;
    //	s.feedtm = false;
    //	s.feedtf = false;
    //	s.xenon = XEType::XE_TR;
    //
    //	simon.runSteady(s);
    //
    //
    //	exit(0);
    //
    //	//for (int idep = 1; idep < simon.nburn(); idep++)
    //	for (int idep = 1; idep < simon.nburn(); idep++)
    //	{
    //		float burn = simon.burn(idep);
    //		//simon.setBurnup(burn);
    //		burn = simon.burn(idep); // MWD/MTU
    //		d_option.tsec = simon.dburn(idep) / simon.pload() * simon.d().totmass() * 3600.0 * 24.0;
    //		simon.runDepletion(d_option);
    //		s.ppm = simon.ppm();
    //		simon.updateBurnup();
    //		simon.runSteady(s);
    //		printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.6f/n", idep+1, burn, simon.ppm(), simon.eigv());
    //
    //	}
    //
    //	//simon.updateBurnup();
    //	//simon.runSteady(s);
    //	//printf("DEPLETION : %d,  BURNUP : %.2f (MWD/MTU), CBC : %.2f (PPM), EIGV : %.6f/n", simon.nburn(), burn, simon.ppm(), simon.eigv());
    //
    //	auto end = chrono::steady_clock::now();
    //	std::cout << "Elapsed time in milliseconds : "
    //		<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
    //		<< " ms" << endl;
    //
    //	//    for (int l = 0; l < simon.g().nxyz(); ++l) {
    //	//            printf("POWER : %e/n", simon.power(l));
    //	//    }
    //
    //	//    for (int idep = 0; idep < 20; idep++)
    //	//    {
    //	//        simon.runKeff(100);
    //	//        simon.runECP(100, 1.0);
    //	//        simon.runDepletion(100);
    //	//        printf("DEPLETION : %d,  CBC : %.2f/n", idep, simon.ppm());
    //	//    }
    //
    //	//    for (int l = 0; l < simon.g().nxyz(); ++l) {
    //	//        for (int ig = 0; ig < simon.g().ng(); ++ig) {
    //	//            printf("FLUX : %e/n", simon.flux(ig,l)*simon.fnorm());
    //	//        }
    //	//    }
    //
    //		//GeometryCuda* g_cuda = new GeometryCuda(simon.g());
    //		//CrossSectionCuda* x_cuda = new CrossSectionCuda(simon.x());
    //		//x_cuda->updateXS(x_cuda->ddmaca(), x_cuda->ddmaca(), x_cuda->ddmaca(), x_cuda->ddmaca());
    //		//test<<<1,1>>>(g_cuda);
    //		//checkCudaErrors(cudaDeviceSynchronize());
    //}
    //
    //// function to call if operator new can't allocate enough memory or error arises
    // void outOfMemHandler() {
    //	std::cerr << "Unable to satisfy request for memory/n";
    //
    //	std::exit(-1);
}

// extern "C" {
// void opendb(int* length, const char* file);
// void readDimension(int* ng, int* nxy, int* nz, int* nx, int* ny, int* nsurf);
// void readIndex(int* nx, int* ny, int* nxy, int* nz, int* nxs, int* nxe, int* nys, int* nye, int* ijtol, int* neibr,
//                float* hmesh);
// void readBoundary(int* symopt, int* symang, float* albedo);
// void readNXNY(const int* nx, const int* ny, float* val);
// void readNXYZ(const int* nxyz, float* val);
//
// void readStep(float* bucyc, float* buavg, float* efpd);
// void readXS(const int* niso, float* xs);
// void readXSS(const int* niso, float* xs);
// void readXSD(const int* niso, float* xs);
// void readXSSD(const int* niso, float* xs);
// void readXSDTM(const int* niso, float* xs);
// void readXSSDTM(const int* niso, float* xs);
// void readDensity(const int* niso, float* dnst);
//
// }

// int main() {
//
//////  1 3 5
//////  3 2 2
//////  5 2 4
//    int n = 3;
//    int _idx_col[] {0,1,2,0,1,2,0,1,2};
//    int _idx_row[] {0,0,0,1,1,1,2,2,2};
//    int _a[] {1,3,5,3,2,2,5,2,4};
//
////    int n = 2;
////    int _idx_col[]{0,1,0,1};
////    int _idx_row[]{0,0,1,1};
////    int _a[]{1,0,0,1};
//    using ValueType = double;
//    using RealValueType = gko::remove_complex<ValueType>;
//    using IndexType = int;
//
//    using vec = gko::matrix::Dense<ValueType>;
//    using real_vec = gko::matrix::Dense<RealValueType>;
//    using mtx = gko::matrix::Dense<ValueType>;
//    using cg = gko::solver::Cg<ValueType>;
//    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
//
//    // executor where Ginkgo will perform the computation
////    const auto exec = gko::OmpExecutor::create();
//    const auto exec = gko::ReferenceExecutor::create();
//
//    auto matrix = share(mtx::create(exec, gko::dim<2>(n)));
//    for (int i = 0; i < n*n; ++i) {
//        matrix->at(_idx_row[i], _idx_col[i]) = _a[i];
//    }
//
//
//    auto rhs = vec::create(exec, gko::dim<2>(n, 1));
//    for (int i = 0; i < n; ++i) {
//        rhs->at(i) = 1.0;
//    }
//
//    auto u = vec::create(exec, gko::dim<2>(n, 1));
//
//    unsigned int nmaxiter = 100;
//
//    using bj = gko::preconditioner::Jacobi<>;
//    auto solver_factory = cg::build()
//            .with_criteria(gko::stop::Iteration::build()
//                                   .with_max_iters(nmaxiter)
//                                   .on(exec),
//                           gko::stop::ResidualNormReduction<>::build()
//                                   .with_reduction_factor(1e-6)
//                                   .on(exec))
////            .with_preconditioner(bj::build().with_max_block_size(8u).on(exec))
//            .on(exec);
//
//    auto solver = solver_factory->generate(matrix);
//    solver->apply(gko::lend(rhs), gko::lend(u));
//
//    for (int i = 0; i < n; ++i) {
//        printf("%f/n", u->at(i));
//    }
//
//    for (int i = 0; i < n; ++i) {
//        rhs->at(i) = 2.0;
//    }
//
//    solver->apply(gko::lend(rhs), gko::lend(u));
//
//    for (int i = 0; i < n; ++i) {
//        printf("%f/n", u->at(i));
//    }
//
//}

// int main() {
//
//     for (int i = 0; i < 100; ++i) {
//         printf("%d",i);
//     }
//     exit(-1);

//    int ng;
//    int nx;
//    int ny;
//    int nz;
//    int nxy;
//    int nxyz;
//    int nsurf;
//
//    //string simondb = "/Users/jiyoon/Downloads/simondb0";
//    string simondb = "simondb0";
//    int length = simondb.length();
//    printf("Compledted to open Simon DB File./n");
//    opendb(&length, simondb.c_str());
//    readDimension(&ng, &nxy, &nz, &nx, &ny, &nsurf);
//    nxyz = nxy * nz;
//    nsurf = nsurf * nz + (nz + 1) * nxy;
//
//    int* nxs = new int[ny];
//    int* nxe = new int[ny];
//    int* nys = new int[nx];
//    int* nye = new int[nx];
//    int* ijtol = new int[nx * ny];
//    int* neibr = new int[NEWS * nxy];
//    float* hmesh = new float[NDIRMAX * nxyz];
//    float* chflow = new float[nx * ny];
//
//
//    readIndex(&nx, &ny, &nxy, &nz, nxs, nxe, nys, nye, ijtol, neibr, hmesh);
//
//    int symopt;
//    int symang;
//    float albedo[6];
//
//    readBoundary(&symopt, &symang, albedo);
//
//    Geometry* g = new Geometry();
//    g->initDimension(&ng, &nxy, &nz, &nx, &ny, &nsurf);
//    g->initIndex(nxs, nxe, nys, nye, ijtol, neibr, hmesh);
//    g->setBoudnaryCondition(&symopt, &symang, albedo);
//
//    int NPTM = 2;
//    SteamTable steam;
//    CrossSection* x = new CrossSection(ng, NISO, NFIS, NNIS, NPTM, nxyz);
//    DepletionChain* d = new DepletionChain(*g);
//    Feedback* f = new Feedback(*g, steam);
//
//    readNXNY(&nx, &ny, chflow);
//    readNXYZ(&nxyz, &(f->ppm0(0)));
//    readNXYZ(&nxyz, &(f->stf0(0)));
//    readNXYZ(&nxyz, &(f->tm0(0)));
//    readNXYZ(&nxyz, &(f->dm0(0)));
//
//
//    BICGCMFD cmfd(*g, *x);
//    cmfd.setNcmfd(5);
//    cmfd.setEpsl2(1.0E-7);
//    cmfd.setEshift(0.04);
//
//    NodalCPU nodal(*g, *x);
//
//    float* power = new float[nxyz];
//
//    int nstep = 1;
//
//    for (int istep = 0; istep < nstep; istep++) {
//        float bucyc, buavg, efpd;
//        readStep(&bucyc, &buavg, &efpd);
//        readDensity(&NISO, &(d->dnst(0, 0)));
//        readNXYZ(&nxyz, &(d->burn(0)));
//        readNXYZ(&nxyz, power);
//        readNXYZ(&nxyz, &(f->tf(0)));
//        readNXYZ(&nxyz, &(f->tm(0)));
//        readNXYZ(&nxyz, &(f->dm(0)));
//
//        readXS(&NISO, &(x->xsmicd0(0, 0, 0)));
//        readXS(&NISO, &(x->xsmica0(0, 0, 0)));
//        readXS(&NISO, &(x->xsmicf0(0, 0, 0)));
//        readXS(&NISO, &(x->xsmicn0(0, 0, 0)));
//        readXS(&NISO, &(x->xsmick0(0, 0, 0)));
//        readXSS(&NISO, &(x->xsmics0(0, 0, 0, 0)));
//
//        readXSD(&NISO, &(x->xdpmicd(0, 0, 0)));
//        readXSD(&NISO, &(x->xdpmica(0, 0, 0)));
//        readXSD(&NISO, &(x->xdpmicf(0, 0, 0)));
//        readXSD(&NISO, &(x->xdpmicn(0, 0, 0)));
//        readXSD(&NISO, &(x->xdpmick(0, 0, 0)));
//        readXSSD(&NISO, &(x->xdpmics(0, 0, 0, 0)));
//
//        readXSD(&NISO, &(x->xdfmicd(0, 0, 0)));
//        readXSD(&NISO, &(x->xdfmica(0, 0, 0)));
//        readXSD(&NISO, &(x->xdfmicf(0, 0, 0)));
//        readXSD(&NISO, &(x->xdfmicn(0, 0, 0)));
//        readXSD(&NISO, &(x->xdfmick(0, 0, 0)));
//        readXSSD(&NISO, &(x->xdfmics(0, 0, 0, 0)));
//
//        readXSD(&NISO, &(x->xddmicd(0, 0, 0)));
//        readXSD(&NISO, &(x->xddmica(0, 0, 0)));
//        readXSD(&NISO, &(x->xddmicf(0, 0, 0)));
//        readXSD(&NISO, &(x->xddmicn(0, 0, 0)));
//        readXSD(&NISO, &(x->xddmick(0, 0, 0)));
//        readXSSD(&NISO, &(x->xddmics(0, 0, 0, 0)));
//
//
//        readXSDTM(&NISO, &(x->xdmmicd(0, 0, 0, 0)));
//        readXSDTM(&NISO, &(x->xdmmica(0, 0, 0, 0)));
//        readXSDTM(&NISO, &(x->xdmmicf(0, 0, 0, 0)));
//        readXSDTM(&NISO, &(x->xdmmicn(0, 0, 0, 0)));
//        readXSDTM(&NISO, &(x->xdmmick(0, 0, 0, 0)));
//        readXSSDTM(&NISO, &(x->xdmmics(0, 0, 0, 0, 0)));
//
//    }
//
//    float ppm = 800.0;
//
//    x->updateMacroXS(&(d->dnst(0, 0)));
//    f->initDelta(ppm);
//    x->updateXS(&(d->dnst(0, 0)), &(f->dppm(0)), &(f->dtf(0)), &(f->dtm(0)));
//
//
//    double eigv = 1.0;
//    float errl2 = 1.0;
//    double* phif = new double[g->ngxyz()]{};
//    std::fill_n(phif,g->ngxyz(), 1.0);
//
//    cmfd.setEshift(0.01);
//
//    cmfd.updpsi(phif);
//    cmfd.upddtil();
//    cmfd.setls(eigv);
//    cmfd.drive(eigv, phif, errl2);
//
//
//    double ppmd = ppm;
//    double eigvd = eigv;
//
//    ppm = ppm + (eigv-1) * 1E5 / 10.0;
////    printf("EIGV : %10.6f with PPM %f10.3 --> ESTIMATING PPM : %f10.3/n", reigv, ppmd, ppm);
////    f->updatePPM(ppm);
////    x->updateXS(&(d->dnst(0, 0)), &(f->dppm(0)), &(f->dtf(0)), &(f->dtm(0)));
////    cmfd.upddtil();
////    cmfd.setls(eigv);
////    cmfd.drive(eigv, phif, errl2);
//
//
//    double eigvt = 1.0;
//    for (int iout = 0; iout < 100; ++iout) {
//        printf("EIGV : %10.6f with PPM %f10.3 --> ESTIMATING PPM : %f10.3/n", eigv, ppmd, ppm);
//
//        f->updatePPM(ppm);
//        d->updateH2ODensity(&f->dm(0), ppm);
//        x->updateXS(&(d->dnst(0, 0)), &(f->dppm(0)), &(f->dtf(0)), &(f->dtm(0)));
//        cmfd.upddtil();
//        cmfd.setls(eigv);
//        cmfd.drive(eigv, phif, errl2);
//
//        if(abs(eigv-eigvt) < 1.E-6) break;
//
//        double temp =ppm;
//        ppm = (ppm - ppmd)/(eigv - eigvd)  *(eigvt-eigv) + ppm;
//        ppmd = temp;
//        eigvd = eigv;
//
//    }
//
//
//    //int maxout = 1;
//
//    //for (int iout = 0; iout < maxout; iout++)
//    //{
//    //    float dppm = 0.0;
//
//    //    x->updateXS(&(d->dnst(0, 0)), dppm, &(f->dtf(0)), &(f->dtm(0)));
//    //    cmfd.upddtil();
//
//    //    cmfd.setls();
//    //    //cmfd.drive(reigv, phif, psi, errl2);
//    //    //cmfd.updjnet(phif, jnet);
//    //    //nodal.reset(xs, reigv, jnet, phif);
//    //    //nodal.drive(jnet);
//    //    //cmfd.upddhat(phif, jnet);
//    //    //cmfd.updjnet(phif, jnet);
//
//    //    //if (iout > 3 && errl2 < 1E-6) break;
//    //}
//
//
//
//    delete[] nxs;
//    delete[] nxe;
//    delete[] nys;
//    delete[] nye;
//    delete[] ijtol;
//    delete[] neibr;
//    delete[]  hmesh;
//
//    delete g;
//    delete x;
//
//
////    CMFDCPU cmfd(_g, xs);
////    cmfd.setNcmfd(7);
////    cmfd.setEpsl2(1.0E-7);
////    cmfd.setEshift(0.00);
////
////    NodalCPU nodal(_g, xs);
////    cmfd.upddtil();
////
////    for (int i = 0; i < 50; ++i) {
////        cmfd.setls();
////        cmfd.drive(reigv, phif, psi, errl2);
////        cmfd.updjnet(phif, jnet);
////        nodal.reset(xs, reigv, jnet, phif);
////        nodal.drive(jnet);
////        cmfd.upddhat(phif, jnet);
////        cmfd.updjnet(phif, jnet);
//////        if (i > 3 && errl2 < 1E-6) break;
////    }
//
////    BICGCMFD bcmfd(g,xs);
////    bcmfd.setNcmfd(100);
////    bcmfd.setEpsl2(1.0E-7);
////
////    bcmfd.upddtil();
////    bcmfd.setls();
////
////    double eigv = 1.0;
////    auto begin = clock();
////    bcmfd.drive(eigv, phif, psi, errl2);
////    auto end = clock();
////    auto elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
////    printf("TIME : %7.3f/n", elapsed_secs);
////
//////    for (int i = 0; i < 50; ++i) {
//////        bcmfd.drive(reigv, phif, psi, errl2);
////////        cmfd.updjnet(phif, jnet);
////////        nodal.reset(xs, reigv, jnet, phif);
////////        nodal.drive(jnet);
////////        cmfd.upddhat(phif, jnet);
////////        cmfd.updjnet(phif, jnet);
//////
//////    }
////
////
////    return 0;
//}