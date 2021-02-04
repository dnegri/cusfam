#include "pch.h"
#include <time.h>
#include "NodalCPU.h"
#include "CMFDCPU.h"
#include "BICGCMFD.h"
#include "Geometry.h"
#include "DepletionChain.h"
#include "CrossSection.h"
#include "Feedback.h"


// function to call if operator new can't allocate enough memory or error arises
void outOfMemHandler()
{
    std::cerr << "Unable to satisfy request for memory\n";

    std::exit(-1);
}

extern"C" {
    void opendb(int* length, const char* file);
    void readDimension(int* ng, int* nxy, int* nz, int* nx, int* ny, int* nsurf);
    void readIndex(int* nx, int* ny, int* nxy, int* nz, int* nxs, int* nxe, int* nys, int* nye, int* ijtol, int* neibr, float* hmesh);
    void readBoundary(int* symopt, int* symang, float* albedo);
    void readNXNY(const int* nx, const int* ny, float* val);
    void readNXYZ(const int* nxyz, float* val);

    void readStep(float* bucyc, float* buavg, float* efpd);
    void readXS(const int* niso, float* xs);
    void readXSS(const int* niso, float* xs);
    void readXSD(const int* niso, float* xs);
    void readXSSD(const int* niso, float* xs);
    void readXSDTM(const int* niso, float* xs);
    void readXSSDTM(const int* niso, float* xs);
    void readDensity(const int* niso, float* dnst);

}


int main() {

    int ng;
    int nx;
    int ny;
    int nz;
    int nxy;
    int nxyz;
    int nsurf;

    string simondb = "simondb0";
    int length = simondb.length();
    opendb(&length, simondb.c_str());
    readDimension(&ng, &nxy, &nz, &nx, &ny, &nsurf);
    nxyz = nxy * nz;
    nsurf = nsurf * nz + (nz + 1) * nxy;

    int* nxs = new int[ny];
    int* nxe = new int[ny];
    int* nys = new int[nx];
    int* nye = new int[nx];
    int* ijtol = new int[nx * ny];
    int* neibr = new int[NEWS * nxy];
    float* hmesh = new float[NDIRMAX * nxyz];
    float* chflow = new float[nx*ny];


    readIndex(&nx, &ny, &nxy, &nz, nxs, nxe, nys, nye, ijtol, neibr, hmesh);

    int symopt;
    int symang;
    float albedo[6];

    readBoundary(&symopt, &symang, albedo);

    Geometry* g = new Geometry();
    g->initDimension(&ng, &nxy, &nz, &nx, &ny, &nsurf);
    g->initIndex(nxs, nxe, nys, nye, ijtol, neibr, hmesh);
    g->setBoudnaryCondition(&symopt, &symang, albedo);

    int NPTM = 2;
    SteamTable steam;
    CrossSection* x = new CrossSection(ng, NISO, NFIS, NNIS, NPTM, nxyz);
    DepletionChain* d = new DepletionChain(*g);
    Feedback* f = new Feedback(*g, steam);
    
    readNXNY(&nx, &ny, chflow);
    readNXYZ(&nxyz, &(f->ppm0(0)));
    readNXYZ(&nxyz, &(f->stf0(0)));
    readNXYZ(&nxyz, &(f->tm0(0)));
    readNXYZ(&nxyz, &(f->dm0(0)));


    CMFDCPU cmfd(*g, *x);
    cmfd.setNcmfd(7);
    cmfd.setEpsl2(1.0E-7);
    cmfd.setEshift(0.00);

    NodalCPU nodal(*g, *x);

    float* power = new float[nxyz];

    int nstep = 1;

    for (size_t istep = 0; istep < nstep; istep++)
    {
        float bucyc, buavg, efpd;
        readStep(&bucyc, &buavg, &efpd);
        readDensity(&NISO, &(d->dnst(0, 0)));
        readNXYZ(&nxyz, &(d->burn(0)));
        readNXYZ(&nxyz, power);
        readNXYZ(&nxyz, &(f->tf(0)));
        readNXYZ(&nxyz, &(f->tm(0)));
        readNXYZ(&nxyz, &(f->dm(0)));

        readXS(&NISO, &(x->xsmicd0(0, 0, 0)));
        readXS(&NISO, &(x->xsmica0(0, 0, 0)));
        readXS(&NISO, &(x->xsmicf0(0, 0, 0)));
        readXS(&NISO, &(x->xsmicn0(0, 0, 0)));
        readXS(&NISO, &(x->xsmick0(0, 0, 0)));
        readXSS(&NISO, &(x->xsmics0(0, 0, 0, 0)));

        readXSD(&NISO, &(x->xdpmicd(0, 0, 0)));
        readXSD(&NISO, &(x->xdpmica(0, 0, 0)));
        readXSD(&NISO, &(x->xdpmicf(0, 0, 0)));
        readXSD(&NISO, &(x->xdpmicn(0, 0, 0)));
        readXSD(&NISO, &(x->xdpmick(0, 0, 0)));
        readXSSD(&NISO, &(x->xdpmics(0, 0, 0, 0)));

        readXSD(&NISO, &(x->xdfmicd(0, 0, 0)));
        readXSD(&NISO, &(x->xdfmica(0, 0, 0)));
        readXSD(&NISO, &(x->xdfmicf(0, 0, 0)));
        readXSD(&NISO, &(x->xdfmicn(0, 0, 0)));
        readXSD(&NISO, &(x->xdfmick(0, 0, 0)));
        readXSSD(&NISO, &(x->xdfmics(0, 0, 0, 0)));

        readXSD(&NISO, &(x->xddmicd(0, 0, 0)));
        readXSD(&NISO, &(x->xddmica(0, 0, 0)));
        readXSD(&NISO, &(x->xddmicf(0, 0, 0)));
        readXSD(&NISO, &(x->xddmicn(0, 0, 0)));
        readXSD(&NISO, &(x->xddmick(0, 0, 0)));
        readXSSD(&NISO,&(x->xddmics(0, 0, 0, 0)));


        readXSDTM (&NISO, &(x->xdmmicd(0, 0, 0, 0)));
        readXSDTM(&NISO, &(x->xdmmica(0, 0, 0, 0)));
        readXSDTM(&NISO, &(x->xdmmicf(0, 0, 0, 0)));
        readXSDTM(&NISO, &(x->xdmmicn(0, 0, 0, 0)));
        readXSDTM(&NISO, &(x->xdmmick(0, 0, 0, 0)));
        readXSSDTM(&NISO, &(x->xdmmics(0, 0, 0, 0, 0)));

    }


    x->updateMacroXS(&(d->dnst(0, 0)));
    x->updateXS(&(d->dnst(0, 0)), 0.0, &(f->dtf(0)), &(f->dtm(0)));
    
    cmfd.upddtil();
    cmfd.setls();

    double reigv = 1.0;
    float errl2 = 1.0;
    double* phif = new double[nxyz * ng]{};
    double* psi = new double[nxyz]{};

    for (int l = 0; l < nxyz; l++)
    {
        for (int ig = 0; ig < ng; ig++)
        {
            phif[ig + l * ng] = 1.0;
        }

        psi[l] = 0.0;

        for (int ig = 0; ig < ng; ig++)
        {
            psi[l] = phif[ig + l * ng] * x->xsnf(ig, l);
        }
        psi[l] = psi[l] * g->vol(l);

    }

    cmfd.drive(reigv, phif, psi, errl2);


    //int maxout = 1;

    //for (size_t iout = 0; iout < maxout; iout++)
    //{
    //    float dppm = 0.0;

    //    x->updateXS(&(d->dnst(0, 0)), dppm, &(f->dtf(0)), &(f->dtm(0)));
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



    delete [] nxs;
    delete [] nxe;
    delete [] nys;
    delete [] nye;
    delete [] ijtol;
    delete [] neibr;
    delete []  hmesh;

    delete g;
    delete x;


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