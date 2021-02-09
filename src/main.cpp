#include "pch.h"
#include <time.h>
#include "NodalCPU.h"
#include "CMFDCPU.h"
#include "BICGCMFD.h"
#include "Geometry.h"
#include "GeometryCuda.h"
#include "DepletionChain.h"
#include "CrossSection.h"
#include "CrossSectionCuda.h"
#include "Feedback.h"
#include "omp.h"
#include "SimonCPU.h"

#ifndef CPU
dim3 BLOCKS_NODE;
dim3 THREADS_NODE;
dim3 BLOCKS_SURFACE;
dim3 THREADS_SURFACE;
#endif


__global__ void test(void* a)
{
    printf("CUDA : %f\n", ((Geometry*)a)->albedo(1, 1));
}

int main() {
    SimonCPU simon;
    simon.initialize("../run/simondb0");

    #ifndef CPU
    BLOCKS_NODE = dim3(simon.g().nxyz() / NTHREADSPERBLOCK + 1, 1, 1);
    THREADS_NODE = dim3(NTHREADSPERBLOCK, 1, 1);
    BLOCKS_SURFACE = dim3(simon.g().nsurf() / NTHREADSPERBLOCK + 1, 1, 1);
    THREADS_SURFACE = dim3(NTHREADSPERBLOCK, 1, 1);
#endif

    simon.setBurnup(1000);
    //simon.runKeff(100);
    simon.runECP(100, 1.0);
    //GeometryCuda* g_cuda = new GeometryCuda(simon.g());
    //CrossSectionCuda* x_cuda = new CrossSectionCuda(simon.x());
    //x_cuda->updateXS(x_cuda->ddmaca(), x_cuda->ddmaca(), x_cuda->ddmaca(), x_cuda->ddmaca());
    //test<<<1,1>>>(g_cuda);
    //cudaDeviceSynchronize();
}

// function to call if operator new can't allocate enough memory or error arises
void outOfMemHandler() {
    std::cerr << "Unable to satisfy request for memory\n";

    std::exit(-1);
}



//extern "C" {
//void opendb(int* length, const char* file);
//void readDimension(int* ng, int* nxy, int* nz, int* nx, int* ny, int* nsurf);
//void readIndex(int* nx, int* ny, int* nxy, int* nz, int* nxs, int* nxe, int* nys, int* nye, int* ijtol, int* neibr,
//               float* hmesh);
//void readBoundary(int* symopt, int* symang, float* albedo);
//void readNXNY(const int* nx, const int* ny, float* val);
//void readNXYZ(const int* nxyz, float* val);
//
//void readStep(float* bucyc, float* buavg, float* efpd);
//void readXS(const int* niso, float* xs);
//void readXSS(const int* niso, float* xs);
//void readXSD(const int* niso, float* xs);
//void readXSSD(const int* niso, float* xs);
//void readXSDTM(const int* niso, float* xs);
//void readXSSDTM(const int* niso, float* xs);
//void readDensity(const int* niso, float* dnst);
//
//}

//int main() {
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
//        printf("%f\n", u->at(i));
//    }
//
//    for (int i = 0; i < n; ++i) {
//        rhs->at(i) = 2.0;
//    }
//
//    solver->apply(gko::lend(rhs), gko::lend(u));
//
//    for (int i = 0; i < n; ++i) {
//        printf("%f\n", u->at(i));
//    }
//
//}



//int main() {
//
//#pragma omp parallel for
//    for (int i = 0; i < 100; ++i) {
//        printf("%d",i);
//    }
//    exit(-1);

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
//    printf("Compledted to open Simon DB File.\n");
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
//    for (size_t istep = 0; istep < nstep; istep++) {
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
////    printf("EIGV : %10.6f with PPM %f10.3 --> ESTIMATING PPM : %f10.3\n", reigv, ppmd, ppm);
////    f->updatePPM(ppm);
////    x->updateXS(&(d->dnst(0, 0)), &(f->dppm(0)), &(f->dtf(0)), &(f->dtm(0)));
////    cmfd.upddtil();
////    cmfd.setls(eigv);
////    cmfd.drive(eigv, phif, errl2);
//
//
//    double eigvt = 1.0;
//    for (int iout = 0; iout < 100; ++iout) {
//        printf("EIGV : %10.6f with PPM %f10.3 --> ESTIMATING PPM : %f10.3\n", eigv, ppmd, ppm);
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
//    //for (size_t iout = 0; iout < maxout; iout++)
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
////    printf("TIME : %7.3f\n", elapsed_secs);
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