#include "pch.h"
#include "NodalCPU.h"

int main() {

    int ng=2;
    int nx=2;
    int ny=2;
    int nz=1;
    int nxy=nx*ny;
    int nxyz = nxy * nz;
    int lsurf = 12;

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

    double* hmesh = new double[(NDIRMAX+1)*nxyz]{ 20.87562, 20.87562, 20.87562, 381.0000, 20.87562, 20.87562, 20.87562, 381.0000, 20.87562, 20.87562, 20.87562, 381.0000, 20.87562, 20.87562, 20.87562, 381.0000 };
    double* jnet = new double[LR * ng * NDIRMAX * nxyz]{};
    double * phif  = new double[ng*nxyz]{0.803122708205631,0.164597288693038,0.802536075765036,0.175546129329372,0.802536062402559,0.175546126476501,0.803122301848942,0.164597207588429};


    for (int l = 0; l < nxyz; ++l) {
        for (int idir = 0; idir < NDIRMAX; ++idir) {
            hmesh[l*(NDIRMAX+1)+idir+1] = 1.0;
            for (int ig = 0; ig < ng; ++ig) {
                *jnet++ = 0.0;
                *jnet++ = 0.0;
            }
        }

    jnet = jnet-LR*ng*NDIRMAX*nxyz;
    phif = phif-ng*nxyz;

    Geometry g;
    g.init(&ng, &nxy, &nz, &nx, &ny, nxs, nxe, nys, nye, &lsurf, ijtol, neibr, hmesh);
    CrossSection xs(ng, nxyz);

    NodalCPU cpu(g, xs);

    cpu.init();
    cpu.reset(xs, &reigv, jnet, phif);
    cpu.drive();




    return 0;
}
