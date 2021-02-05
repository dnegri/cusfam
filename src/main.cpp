#include "pch.h"
#include <time.h>
#include "NodalCPU.h"
#include "CMFDCPU.h"
#include "BICGCMFD.h"
#include "Geometry.h"
#include "DepletionChain.h"
#include "CrossSection.h"
#include "Feedback.h"
#include "Simon.h"


// function to call if operator new can't allocate enough memory or error arises
void outOfMemHandler()
{
    std::cerr << "Unable to satisfy request for memory\n";

    std::exit(-1);
}

int main() {

    Simon simon;

    //string simondb = "/Users/jiyoon/Downloads/simondb";
    string simondb = "simondb";
    simon.initialize(simondb.c_str());

    simon.setBurnup(0);
    simon.runStatic(100);

    //int maxout = 1;

    //for (size_t iout = 0; iout < maxout; iout++)
    //{
    //    float dppm = 0.0;

    //    x.updateXS(&(d.dnst(0, 0)), dppm, &(f.dtf(0)), &(f.dtm(0)));
    //    cmfd.upddtil();

    //    cmfd.setls();
    //    //cmfd.drive(reigv, phif, psi, errl2);
    //    //cmfd.updjnet(phif, jnet);
    //    //nodal.reset(xs, reigv, jnet, phif);
    //    //nodal.drive(jnet);
    //    //cmfd.upddhat(phif, jnet);
    //    //cmfd.updjnet(phif, jnet);

    //    //if (iout > 3 && errl2 < 1E-6) break;
    //}





//    CMFDCPU cmfd(_g, xs);
//    cmfd.setNcmfd(7);
//    cmfd.setEpsl2(1.0E-7);
//    cmfd.setEshift(0.00);
//
//    NodalCPU nodal(_g, xs);
//    cmfd.upddtil();
//
//    for (int i = 0; i < 50; ++i) {
//        cmfd.setls();
//        cmfd.drive(reigv, phif, psi, errl2);
//        cmfd.updjnet(phif, jnet);
//        nodal.reset(xs, reigv, jnet, phif);
//        nodal.drive(jnet);
//        cmfd.upddhat(phif, jnet);
//        cmfd.updjnet(phif, jnet);
////        if (i > 3 && errl2 < 1E-6) break;
//    }

//    BICGCMFD bcmfd(g,xs);
//    bcmfd.setNcmfd(100);
//    bcmfd.setEpsl2(1.0E-7);
//
//    bcmfd.upddtil();
//    bcmfd.setls();
//
//    double eigv = 1.0;
//    auto begin = clock();
//    bcmfd.drive(eigv, phif, psi, errl2);
//    auto end = clock();
//    auto elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    printf("TIME : %7.3f\n", elapsed_secs);
//
////    for (int i = 0; i < 50; ++i) {
////        bcmfd.drive(reigv, phif, psi, errl2);
//////        cmfd.updjnet(phif, jnet);
//////        nodal.reset(xs, reigv, jnet, phif);
//////        nodal.drive(jnet);
//////        cmfd.upddhat(phif, jnet);
//////        cmfd.updjnet(phif, jnet);
////
////    }
//
//
//    return 0;
}