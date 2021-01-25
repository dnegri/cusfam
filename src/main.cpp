#include "pch.h"
#include "NodalCPU.h"
#include "CMFDCPU.h"

int main() {

    int ng=2;
    int nx=2;
    int ny=2;
    int nz=2;
    int nxy=nx*ny;
    int nxyz = nxy * nz;
    int lsurf = 20;

    int* nxs = new int[ny]{1,1};
    int* nxe = new int[ny]{2,2};
    int* nys = new int[nx]{1,1};
    int* nye = new int[nx]{2,2};
    int* ijtol = new int[nx*ny]{};
    ijtol[0] = 1;ijtol[1] = 2;ijtol[2] = 3;ijtol[3] = 4;
    int* neibr = new int[NEWS*nxy];
    neibr[0]=0;neibr[1]=2;neibr[2]=0;neibr[3]=3;
    neibr[4]=1;neibr[5]=0;neibr[6]=0;neibr[7]=4;
    neibr[8]=0;neibr[9]=4;neibr[10]=1;neibr[11]=0;
    neibr[12]=3;neibr[13]=0;neibr[14]=2;neibr[15]=0;
    double reigv = 0.742138228032457;

    double* hmesh = new double[(NDIRMAX+1)*nxyz]{ 
        20.87562, 20.87562, 20.87562, 38.1, 20.87562, 20.87562, 20.87562, 38.1, 20.87562, 20.87562, 20.87562, 38.1, 20.87562, 20.87562, 20.87562, 38.1,
        20.87562, 20.87562, 20.87562, 38.1, 20.87562, 20.87562, 20.87562, 38.1, 20.87562, 20.87562, 20.87562, 38.1, 20.87562, 20.87562, 20.87562, 38.1
    };
    double* jnet = new double[LR * ng * NDIRMAX * nxyz]{};
    double * phif  = new double[ng*nxyz]{};
    double * psi  = new double[nxyz]{};
    double* albedo = new double[NDIRMAX*LR]{};


    Geometry g;
    g.init(&ng, &nxy, &nz, &nx, &ny, nxs, nxe, nys, nye, &lsurf, ijtol, neibr, hmesh);
    int symang = 360;
    int symopt = 0;
    g.setBoudnaryCondition(&symopt,&symang, albedo);
    CrossSection xs(ng, nxyz);


    for (size_t l = 0; l < nxyz; l++)
    {
        phif[l * ng + 0] = 1.0; //0.803122708205631;
        phif[l * ng + 1] = 1.0; //0.803122708205631;

        xs.xsdf(0, l) = 1.42507149436478;
        xs.xsdf(1, l) = 0.448454988085883;
        xs.xstf(0, l) = 2.592368710035673E-02;
        xs.xstf(1, l) = 8.161927734374642E-02;
        xs.xssf(0, 1, l) = 1.669964031075737E-02;
        xs.xsnf(0, l) = 7.009412307796224E-03;
        xs.xsnf(1, l) = 0.136542037889586;
        xs.chif(0, l) = 1.0;
        xs.chif(1, l) = 0.0;
        xs.xsadf(0, l) = 1.0;
        xs.xsadf(1, l) = 1.0;
    }

    xs.xsnf(0, 0) *= 1.1;
    xs.xsnf(1, 0) *= 1.1;
    xs.xsnf(0, 4) *= 1.1;
    xs.xsnf(1, 4) *= 1.1;
    xs.xstf(0, 0) *= 1.1;
    xs.xstf(1, 0) *= 1.1;
    xs.xstf(0, 4) *= 1.1;
    xs.xstf(1, 4) *= 1.1;

    for (int l = 0; l < nxyz; ++l) {
        psi[l] = (phif[l*ng+0]*xs.xsnf(0,l)+phif[l*ng+1]*xs.xsnf(1,l))*g.vol(l);
    }

    reigv = 1.0;
    float errl2 = 1.0;

    CMFDCPU cmfd(g, xs);
    cmfd.setNcmfd(7);
    cmfd.setEpsl2(1.0E-7);
    cmfd.setEshift(0.000);

    NodalCPU nodal(g, xs);
    cmfd.upddtil();

    for (int i = 0; i < 50; ++i) {
        cmfd.setls();
        cmfd.drive(reigv, phif, psi, errl2);
        cmfd.updjnet(phif, jnet);
        nodal.reset(xs, reigv, jnet, phif);
        nodal.drive(jnet);
        cmfd.upddhat(phif, jnet);
        //if (i > 3 && errl2 < 1E-6) break;
    }


    return 0;
}