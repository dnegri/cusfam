#include "pch.h"
#include "NodalCPU.h"
#include "CMFDCPU.h"
#include "BICGCMFD.h"


// function to call if operator new can't allocate enough memory or error arises
void outOfMemHandler()
{
    std::cerr << "Unable to satisfy request for memory\n";

    std::exit(-1);
}


int main() {


    //set the new_handler
    std::set_new_handler(outOfMemHandler);

    int symang = 360;
    int symopt = 0;
    int ng=2;
    int nx=2;
    int ny=2;
    int nz=1000;
    int nxy=3;
    int nxyz = nxy * nz;
    int lsurf = 5*2;

    int* nxs = new int[ny]{1,1};
    int* nxe = new int[ny]{2,1};
    int* nys = new int[nx]{1,1};
    int* nye = new int[nx]{2,1};
    int* ijtol = new int[nx*ny]{};
    ijtol[0] = 1;ijtol[1] = 2;ijtol[2] = 3;ijtol[3] = 0;
    int* neibr = new int[NEWS*nxy];
    neibr[0]=0;neibr[1]=2;neibr[2]=0;neibr[3]=3;
    neibr[4]=1;neibr[5]=0;neibr[6]=0;neibr[7]=0;
    neibr[8]=0;neibr[9]=0;neibr[10]=1;neibr[11]=0;

    double* hmesh = new double[(NDIRMAX+1)*nxyz];
    for (int l = 0; l < nxyz; ++l) {
        hmesh[l*(NDIRMAX+1)+0] = 20;
        hmesh[l*(NDIRMAX+1)+1] = 20;
        hmesh[l*(NDIRMAX+1)+2] = 20;
        hmesh[l*(NDIRMAX+1)+3] = 5;
    }

    double* jnet = new double[LR * ng * NDIRMAX * nxyz]{};
    double * phif  = new double[ng*nxyz]{};
    double * psi  = new double[nxyz]{};
    double* albedo = new double[NDIRMAX*LR]{};


    Geometry g;
    g.init(&ng, &nxy, &nz, &nx, &ny, nxs, nxe, nys, nye, &lsurf, ijtol, neibr, hmesh);
    g.setBoudnaryCondition(&symopt,&symang, albedo);
    CrossSection xs(ng, nxyz);


    for (size_t l = 0; l < nxyz; l++)
    {
        phif[l * ng + 0] = 1.0; //0.803122708205631;
        phif[l * ng + 1] = 1.0; //0.803122708205631;

        xs.xsdf(0, l) = 1.5;
        xs.xsdf(1, l) = 0.5;
        xs.xstf(0, l) = 0.03;
        xs.xstf(1, l) = 0.08;
        xs.xssf(0, 1, l) = 0.02;
        xs.xsnf(0, l) = 0.001;
        xs.xsnf(1, l) = 0.15;
        xs.chif(0, l) = 1.0;
        xs.chif(1, l) = 0.0;
        xs.xsadf(0, l) = 1.0;
        xs.xsadf(1, l) = 1.0;
    }

    xs.xstf(0, 0) *= 1.1;
    xs.xstf(1, 0) *= 1.1;
//    xs.xstf(0, 3) *= 1.1;
//    xs.xstf(1, 3) *= 1.1;

    for (int l = 0; l < nxyz; ++l) {
        psi[l] = (phif[l*ng+0]*xs.xsnf(0,l)+phif[l*ng+1]*xs.xsnf(1,l))*g.vol(l);
    }

    double reigv = 1.0;
    float errl2 = 1.0;

//    CMFDCPU cmfd(g, xs);
//    cmfd.setNcmfd(7);
//    cmfd.setEpsl2(1.0E-7);
//    cmfd.setEshift(0.00);
//
//    NodalCPU nodal(g, xs);
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

    BICGCMFD bcmfd(g,xs);
    bcmfd.setNcmfd(100);
    bcmfd.setEpsl2(1.0E-7);

    bcmfd.upddtil();
    bcmfd.setls();

    double eigv = 1.0;
    auto begin = clock();
    bcmfd.drive(eigv, phif, psi, errl2);
    auto end = clock();
    auto elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("TIME : %7.3f\n", elapsed_secs);

//    for (int i = 0; i < 50; ++i) {
//        bcmfd.drive(reigv, phif, psi, errl2);
////        cmfd.updjnet(phif, jnet);
////        nodal.reset(xs, reigv, jnet, phif);
////        nodal.drive(jnet);
////        cmfd.upddhat(phif, jnet);
////        cmfd.updjnet(phif, jnet);
//
//    }


    return 0;
}