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
    double * phif  = new double[ng*(nxy+1)*(nz+1)]{};
    double* albedo = new double[NDIRMAX*LR]{};

    phif[1*(nxy+1)*ng + 1*ng + 0] = 0.803122708205631;
    phif[1*(nxy+1)*ng + 1*ng + 1] = 0.164597288693038;
    phif[1*(nxy+1)*ng + 2*ng + 0] = 0.802536075765036;
    phif[1*(nxy+1)*ng + 2*ng + 1] = 0.175546129329372;
    phif[1*(nxy+1)*ng + 3*ng + 0] = 0.802536062402559;
    phif[1*(nxy+1)*ng + 3*ng + 1] = 0.175546126476501;
    phif[1*(nxy+1)*ng + 4*ng + 0] = 0.803122301848942;
    phif[1*(nxy+1)*ng + 4*ng + 1] = 0.164597207588429;

//    jnet[1+1*LR]  = 4.027721933967146E-005;
//    jnet[2+1*LR]  = -2.343435137703359E-004;
//    jnet[1+4*LR]  = -4.025023700280725E-005;
//    jnet[2+4*LR]  = 2.343451886314374E-004;
//    jnet[1+7*LR]  = 4.027813678536138E-005;
//    jnet[2+7*LR]  = -2.343434527089155E-004;
//    jnet[1+10*LR] =  -4.024931955711733E-005;
//    jnet[2+10*LR] =  2.343452496928578E-004;

    Geometry g;
    g.init(&ng, &nxy, &nz, &nx, &ny, nxs, nxe, nys, nye, &lsurf, ijtol, neibr, hmesh);
    int symang = 360;
    int symopt = 0;
    g.setBoudnaryCondition(&symopt,&symang, albedo);
    CrossSection xs(ng, nxyz);

    xs.xsdf(0,0) =	1.42507149436478;
    xs.xsdf(1,0) =	0.448454988085883;
    xs.xsdf(0,1) =	1.44159527781466;
    xs.xsdf(1,1) =	0.445179693901252;
    xs.xsdf(0,2) =	1.44159527781466;
    xs.xsdf(1,2) =	0.445179693901252;
    xs.xsdf(0,3) =	1.42507149436478;
    xs.xsdf(1,3) =	0.448454988085883;

    xs.xstf(0,0) =	2.592368710035673E-02;
    xs.xstf(1,0) =	8.161927734374642E-02;
    xs.xstf(0,1) =	2.653347017658737E-02;
    xs.xstf(1,1) =	7.820114207991198E-02;
    xs.xstf(0,2) =	2.653347017658737E-02;
    xs.xstf(1,2) =	7.820114207991198E-02;
    xs.xstf(0,3) =	2.592368710035673E-02;
    xs.xstf(1,3) =	8.161927734374642E-02;

    xs.xssf(0,0,0) =	0.000000000000000;
    xs.xssf(1,0,0) =	0.000000000000000;
    xs.xssf(0,1,0) =	1.669964031075737E-02;
    xs.xssf(1,1,0) =	0.000000000000000;
    xs.xssf(0,0,1) =	0.000000000000000;
    xs.xssf(1,0,1) =	0.000000000000000;
    xs.xssf(0,1,1) =	1.713363387143419E-02;
    xs.xssf(1,1,1) =	0.000000000000000;
    xs.xssf(0,0,2) =	0.000000000000000;
    xs.xssf(1,0,2) =	0.000000000000000;
    xs.xssf(0,1,2) =	1.713363387143419E-02;
    xs.xssf(1,1,2) =	0.000000000000000;
    xs.xssf(0,0,3) =	0.000000000000000;
    xs.xssf(1,0,3) =	0.000000000000000;
    xs.xssf(0,1,3) =	1.669964031075737E-02;
    xs.xssf(1,1,3) =	0.000000000000000;

    xs.xsnf(0,0) =	7.009412307796224E-03;
    xs.xsnf(1,0) =	0.136542037889586;
    xs.xsnf(0,1) =	7.089077904037993E-03;
    xs.xsnf(1,1) =	0.130755891426204;
    xs.xsnf(0,2) =	7.089077904037993E-03;
    xs.xsnf(1,2) =	0.130755891426204;
    xs.xsnf(0,3) =	7.009412307796224E-03;
    xs.xsnf(1,3) =	0.136542037889586;

    xs.chif(0,0) =	1.0;
    xs.chif(1,0) =	0.0;
    xs.chif(0,1) =	1.0;
    xs.chif(1,1) =	0.0;
    xs.chif(0,2) =	1.0;
    xs.chif(1,2) =	0.0;
    xs.chif(0,3) =	1.0;
    xs.chif(1,3) =	0.0;

    xs.xsadf(0,0) =	1.0;
    xs.xsadf(1,0) =	1.0;
    xs.xsadf(0,1) =	1.0;
    xs.xsadf(1,1) =	1.0;
    xs.xsadf(0,2) =	1.0;
    xs.xsadf(1,2) =	1.0;
    xs.xsadf(0,3) =	1.0;
    xs.xsadf(1,3) =	1.0;

    NodalCPU cpu(g, xs);

    cpu.init();
    cpu.reset(xs, &reigv, jnet, phif);
    cpu.drive();

    return 0;
}