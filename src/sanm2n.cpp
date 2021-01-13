#include "pch.h"
#include "sanm2n.h"

#define jnet(ig,lks)      (jnet[lks*ng + ig])
#define trlcff0(ig,lkd)   (trlcff0[lkd*ng + ig])
#define trlcff1(ig,lkd)   (trlcff1[lkd*ng + ig])
#define trlcff2(ig,lkd)   (trlcff2[lkd*ng + ig])
#define eta1(ig,lkd)  (eta1[lkd*ng + ig])
#define eta2(ig,lkd)  (eta2[lkd*ng + ig])
#define m260(ig,lkd)  (m260[lkd*ng + ig])
#define m251(ig,lkd)  (m251[lkd*ng + ig])
#define m253(ig,lkd)  (m253[lkd*ng + ig])
#define m262(ig,lkd)  (m262[lkd*ng + ig])
#define m264(ig,lkd)  (m264[lkd*ng + ig])
#define diagD(ig,lkd) (diagD[lkd*ng + ig])
#define diagDI(ig,lkd)    (diagDI[lkd*ng + ig])
#define mu(i,j,lkd)   (mu[lkd*ng2 + j*ng + i])
#define tau(i,j,lkd)  (tau[lkd*ng2 + j*ng + i])
#define matM(i,j,lk)  (matM[lk*ng2 + j*ng + i])
#define matMI(i,j,lk) (matMI[lk*ng2 + j*ng + i])
#define matMs(i,j,lk) (matMs[lk*ng2 + j*ng + i])
#define matMf(i,j,lk) (matMf[lk*ng2 + j*ng + i])
#define xssf(i,j,lk)  (xssf[lk*ng2 + j*ng + i])
#define xsadf(ig,lk)  (xsadf[lk*ng + ig])
#define flux(ig,lk)   (flux[lk*ng+ig])

#define dsncff2(ig,lkd) (dsncff2[lkd*ng + ig])
#define dsncff4(ig,lkd) (dsncff4[lkd*ng + ig])
#define dsncff6(ig,lkd) (dsncff6[lkd*ng + ig])

#define hmesh(idir,lk)        (hmesh[lk*NDIRMAX+idir])
#define lktosfc(lr,idir,lk)   (lktosfc[(lk*NDIRMAX+idir)*LR + lr])
#define idirlr(lr,ls)         (idirlr[ls*LR + lr])
#define neib(lr, idir, lk)    (neib[(lk*NDIRMAX+idir)*LR + lr])
#define albedo(lr,idir)       (albedo[idir*LR + lr])

__host__ __device__ void sanm2n_reset(const int& lk,const int& ng,const int& ng2, int& nxyz, float* hmesh, XS_PRECISION* xstf, XS_PRECISION* xsdf, float* eta1, float* eta2, float* m260, float* m251, float* m253, float* m262, float* m264, float* diagD, float* diagDI) {

    int lkd0 = lk * NDIRMAX;
    int lkg0 = lk * ng;

    for (size_t idir = 0; idir < NDIRMAX; idir++)
    {
        int lkd = lkd0 + idir;

        for (size_t ig = 0; ig < ng; ig++)
        {
            int lkg = lkg0 + ig;
            auto kp2 = xstf[lkg] * hmesh(idir, lk) * hmesh(idir, lk) / (4 * xsdf[lkg]);
            auto kp = sqrt(kp2);
            auto kp3 = kp2 * kp;
            auto kp4 = kp2 * kp2;
            auto kp5 = kp2 * kp3;
            auto rkp = 1 / kp;
            auto rkp2 = rkp * rkp;
            auto rkp3 = rkp2 * rkp;
            auto rkp4 = rkp2 * rkp2;
            auto rkp5 = rkp2 * rkp3;
            auto sinhkp = sinh(kp);
            auto coshkp = cosh(kp);

            //calculate coefficient of basic functions P5and P6
            auto bfcff0 = -sinhkp * rkp;
            auto bfcff2 = -5 * (-3 * kp * coshkp + 3 * sinhkp + kp2 * sinhkp) * rkp3;
            auto bfcff4 = -9. * (-105 * kp * coshkp - 10 * kp3 * coshkp + 105 * sinhkp + 45 * kp2 * sinhkp + kp4 * sinhkp) * rkp5;
            auto bfcff1 = -3 * (kp * coshkp - sinhkp) * rkp2;
            auto bfcff3 = -7 * (15 * kp * coshkp + kp3 * coshkp - 15 * sinhkp - 6 * kp2 * sinhkp) * rkp4;

            auto oddtemp = 1 / (sinhkp + bfcff1 + bfcff3);
            auto eventemp = 1 / (coshkp + bfcff0 + bfcff2 + bfcff4);

            //eta1, eta2
            eta1(ig, lkd) = (kp * coshkp + bfcff1 + 6 * bfcff3) * oddtemp;
            eta2(ig, lkd) = (kp * sinhkp + 3 * bfcff2 + 10 * bfcff4) * eventemp;

            //set to variables that depends on node properties by integrating of Pi* pj over - 1 ~1
            m260(ig, lkd) = 2 * eta2(ig, lkd);
            m251(ig, lkd) = 2 * (kp * coshkp - sinhkp + 5 * bfcff3) * oddtemp;
            m253(ig, lkd) = 2 * (kp * (15 + kp2) * coshkp - 3 * (5 + 2 * kp2) * sinhkp) * oddtemp * rkp2;
            m262(ig, lkd) = 2 * (-3 * kp * coshkp + (3 + kp2) * sinhkp + 7 * kp * bfcff4) * eventemp * rkp;
            m264(ig, lkd) = 2 * (-5 * kp * (21 + 2 * kp2) * coshkp + (105 + 45 * kp2 + kp4) * sinhkp) * eventemp * rkp3;
            if (m264(ig, lkd) == 0.0) m264(ig, lkd) = 1.e-10;

            diagD(ig, lkd) = 4 * xsdf[lkg] / (hmesh(idir, lk) * hmesh(idir, lk));
            diagDI(ig, lkd) = 1.0 / diagD(ig, lkd);
        }
    }
}

__host__ __device__ void sanm2n_resetMatrix(const int& lk,const int& ng,const int& ng2, int& nxyz, double& reigv, XS_PRECISION* xstf, XS_PRECISION* xsnff, XS_PRECISION* xschif, XS_PRECISION* xssf, float* matMs, float* matMf, float* matM) {
    int lkg0 = lk * ng;

    for (size_t ige = 0; ige < ng; ige++)
    {
        for (size_t igs = 0; igs < ng; igs++)
        {
            matMs(igs, ige, lk) = -xssf(igs, ige, lk);
            matMf(igs, ige, lk) = xschif[lkg0 + ige] * xsnff[lkg0 + igs];
        }
        matMs(ige, ige, lk) += xstf[lkg0 + ige];

        for (size_t igs = 0; igs < ng; igs++)
        {
            matM(igs, ige, lk) = matMs(igs, ige, lk) - reigv * matMf(igs, ige, lk);
        }
    }

}

__host__ __device__ void sanm2n_prepareMatrix(const int& lk,const int& ng,const int& ng2, int& nxyz, float* m251, float* m253, float* diagD, float* diagDI, float* matM, float* matMI, float* tau, float* mu) {
    int lkd0 = lk * NDIRMAX;

    auto det = matM(0, 0, lk) * matM(1, 1, lk) - matM(1, 0, lk) * matM(0, 1, lk);

    if (abs(det) < 1.E-10) {
        auto rdet = 1 / det;
        matMI(0, 0, lk) = rdet * matM(1, 1, lk);
        matMI(1, 0, lk) = -rdet * matM(1, 0, lk);
        matMI(0, 1, lk) = -rdet * matM(0, 1, lk);
        matMI(1, 1, lk) = rdet * matM(0, 0, lk);
    }
    else {
        matMI(0, 0, lk) = 0;
        matMI(1, 0, lk) = 0;
        matMI(0, 1, lk) = 0;
        matMI(1, 1, lk) = 0;
    }

    auto rm011 = 1. / m011;

    for (size_t idir = 0; idir < NDIRMAX; idir++)
    {
        auto lkd = lkd0 + idir;

        float tempz[2][2] = { 0.0 };

        for (size_t igd = 0; igd < ng; igd++)
        {
            auto tau1 = m033 * (diagDI(igd, lkd) / m253(igd, lkd));

            tempz[igd][igd] = tempz[igd][igd] + m231;

            for (size_t igs = 0; igs < ng; igs++)
            {
                tau(igs, igd, lkd) = tau1 * matM(igs, igd, lk);

                // mu=m011_inv*M_inv*D*(m231*I+m251*tau)
                tempz[igs][igd] += m251(igd, lkd) * tau(igs, igd, lkd);

                // mu=m011_inv*M_inv*D*(m231*I+m251*tau)
                tempz[igs][igd] *= diagD(igd, lkd);
            }
        }

        // mu=m011_inv*M_inv*D*(m231*I+m251*tau)
        mu(0, 0, lkd) = rm011 * (matMI(0, 0, lk) * tempz[0][0] + matMI(1, 0, lk) * tempz[0][1]);
        mu(1, 0, lkd) = rm011 * (matMI(0, 0, lk) * tempz[1][0] + matMI(1, 0, lk) * tempz[1][1]);
        mu(0, 1, lkd) = rm011 * (matMI(0, 1, lk) * tempz[0][0] + matMI(1, 1, lk) * tempz[0][1]);
        mu(1, 1, lkd) = rm011 * (matMI(0, 1, lk) * tempz[1][0] + matMI(1, 1, lk) * tempz[1][1]);

    }

}



__host__ __device__ void sanm2n_trlcffbyintg(float* avgtrl3, float* hmesh3, float& trlcff1, float& trlcff2) {
    float sh[4];

    float rh = (1 / ((hmesh3[LEFT] + hmesh3[CENTER] + hmesh3[RIGHT]) * (hmesh3[LEFT] + hmesh3[CENTER]) * (hmesh3[CENTER] + hmesh3[RIGHT])));
    sh[0] = (2 * hmesh3[LEFT] + hmesh3[CENTER]) * (hmesh3[LEFT] + hmesh3[CENTER]);
    sh[1] = hmesh3[LEFT] + hmesh3[CENTER];
    sh[2] = (hmesh3[CENTER] + 2 * hmesh3[RIGHT]) * (hmesh3[CENTER] + hmesh3[RIGHT]);
    sh[3] = hmesh3[CENTER] + hmesh3[RIGHT];

    if (hmesh3[LEFT] == 0.0) {
        trlcff1 = 0.125 * (5. * avgtrl3[CENTER] + avgtrl3[RIGHT]);
        trlcff2 = 0.125 * (-3. * avgtrl3[CENTER] + avgtrl3[RIGHT]);
    }
    else if (hmesh3[RIGHT] == 0.0) {
        trlcff1 = -0.125 * (5. * avgtrl3[CENTER] + avgtrl3[LEFT]);
        trlcff2 = 0.125 * (-3. * avgtrl3[CENTER] + avgtrl3[LEFT]);
    }
    else {
        trlcff1 = 0.5 * rh * hmesh3[CENTER] * ((avgtrl3[CENTER] - avgtrl3[LEFT]) * sh[2] + (avgtrl3[RIGHT] - avgtrl3[CENTER]) * sh[0]);
        trlcff2 = 0.5 * rh * (hmesh3[CENTER] * hmesh3[CENTER]) * ((avgtrl3[LEFT] - avgtrl3[CENTER]) * sh[3] + (avgtrl3[RIGHT] - avgtrl3[CENTER]) * sh[1]);
    }
}

__host__ __device__ void sanm2n_calculateTransverseLeakage(const int& lk,const int& ng,const int& ng2, int& nxyz, int* lktosfc, int* idirlr, int* neib, float* hmesh, float* albedo, float* jnet, float* trlcff0, float* trlcff1, float* trlcff2)
{
    int lkd0 = lk * NDIRMAX;

    float avgjnet[NDIRMAX];

    for (size_t ig = 0; ig < ng; ig++)
    {
        for (size_t idir = 0; idir < NDIRMAX; idir++)
        {
            auto lsl = lktosfc(LEFT, idir, lk);
            auto lsr = lktosfc(RIGHT, idir, lk);

            avgjnet[idir] = (jnet(ig, lsr) - jnet(ig, lsl)) * hmesh(idir, lk);

            trlcff0(ig, lkd0 + XDIR) = avgjnet[YDIR] + avgjnet[ZDIR];
            trlcff0(ig, lkd0 + YDIR) = avgjnet[XDIR] + avgjnet[ZDIR];
            trlcff0(ig, lkd0 + ZDIR) = avgjnet[XDIR] + avgjnet[YDIR];
        }
    }

    for (size_t idir = 0; idir < NDIRMAX; idir++)
    {
        int lkd = lkd0 + idir;

        for (size_t ig = 0; ig < ng; ig++)
        {
            float avgtrl3[LRC] = { 0.0 };
            float hmesh3[LRC] = { 0.0 };
            hmesh3[CENTER] = hmesh(idir, lk);
            avgtrl3[CENTER] = trlcff0(ig, lkd);

            int lkl = neib(LEFT, idir, lk);
            int lsl = lktosfc(LEFT, idir, lk);
            int idirl = idirlr(LEFT, lsl);

            if (lkl < 0 && albedo(LEFT, idir) == 0) {
                hmesh3[LEFT] = hmesh3[CENTER];
                avgtrl3[LEFT] = avgtrl3[CENTER];
            }
            else {
                hmesh3[LEFT] = hmesh(idirl,lkl);
                avgtrl3[LEFT] = trlcff0(ig, lkd0+idirl);
            }

            int lkr = neib(RIGHT, idir, lk);
            int lsr = lktosfc(RIGHT, idir, lk);
            int idirr = idirlr(RIGHT, lsr);

            if (lkr < 0 && albedo(RIGHT, idir) == 0) {
                hmesh3[RIGHT] = hmesh3[CENTER];
                avgtrl3[RIGHT] = avgtrl3[CENTER];
            }
            else {
                hmesh3[RIGHT] = hmesh(idirr, lkr);
                avgtrl3[RIGHT] = trlcff0(ig, lkd0 + idirr);
            }


            sanm2n_trlcffbyintg(avgtrl3, hmesh3, trlcff1(ig, lkd), trlcff2(ig, lkd));
            //printf("trlcff12: %e %e \n", trlcff1(ig, lkd), trlcff2(ig, lkd));
        }

    }
}

__host__ __device__ void sanm2n_calculateEven(const int& lk,const int& ng,const int& ng2, int& nxyz, float* m260, float* m262, float* m264, float* diagD, float* diagDI, float* matM, double* flux, float* trlcff0, float* trlcff2, float* dsncff2, float* dsncff4, float* dsncff6)
{
    int lkd0 = lk * NDIRMAX;

    for (size_t idir = 0; idir < NDIRMAX; idir++)
    {
        auto lkd = lkd0 + idir;
        float at2[2][2], a[2][2], rm4464[2], bt1[2], bt2[2], b[2];

        for (size_t igd = 0; igd < ng; igd++)
        {
            rm4464[igd] = m044 / m264(igd, lkd);
            auto mu2 = rm4464[igd] * m260(igd, lkd) * diagDI(igd, lkd);

            for (size_t igs = 0; igs < ng; igs++)
            {
                at2[igs][igd] = m022 * rm220 * mu2 * matM(igs, igd, lk);
            }
            at2[igd][igd] += m022 * rm220 * m240;
        }

        printf("at2 : %e %e %e %e\n", at2[0][0], at2[1][0], at2[0][1], at2[1][1]);

        for (size_t igd = 0; igd < ng; igd++)
        {
            auto mu1 = rm4464[igd] * m262(igd, lkd);
            for (size_t igs = 0; igs < ng; igs++)
            {
                a[igs][igd] = mu1 * matM(igs, igd, lk) + matM(0, igd, lk) * at2[igs][0] + matM(1, igd, lk) * at2[igs][1];
            }
            a[igd][igd] += diagD(igd, lkd) * m242;
            bt2[igd] = 2 * (matM(0, igd, lk) * flux(0, lk) + matM(1, igd, lk) * flux(1, lk) + trlcff0(igd, lkd));
            bt1[igd] = m022 * rm220 * diagDI(igd, lkd) * bt2[igd];
        }
        printf("bt1 bt2 : %e %e %e %e\n", bt1[0], bt1[1], bt2[0], bt2[1]);

        for (size_t ig = 0; ig < ng; ig++)
        {
            b[ig] = m220 * trlcff2(ig, lkd) + matM(0, ig, lk) * bt1[0] + matM(1, ig, lk) * bt1[1];
        }

        auto rdet = 1 / (a[0][0] * a[1][1] - a[1][0] * a[0][1]);
        dsncff4(0, lkd) = rdet * (a[1][1] * b[0] - a[1][0] * b[1]);
        dsncff4(1, lkd) = rdet * (a[0][0] * b[1] - a[0][1] * b[0]);

        for (size_t ig = 0; ig < ng; ig++)
        {
            dsncff6(ig, lkd) = diagDI(ig, lkd) * rm4464[ig] * (matM(0, ig, lk) * dsncff4(0, lkd) + matM(1, ig, lk) * dsncff4(1, lkd));
            dsncff2(ig, lkd) = rm220 * (diagDI(ig, lkd) * bt2[ig] - m240 * dsncff4(ig, lkd) - m260(ig, lkd) * dsncff6(ig, lkd));
        }
    }
}

__host__ __device__ void sanm2n_calculateJnet(const int& ls, const int& ng, const int& ng2, int& nsurf, int* lklr, int* idirlr, int* sgnlr, float* albedo, float* hmesh, XS_PRECISION* xsadf, float* m251, float* m253, float* m260, float* m262, float* m264, float* diagD, float* diagDI, float* matM, float* matMI, double* flux, float* trlcff0, float* trlcff1, float* trlcff2, float* mu, float* tau, float* eta1, float* eta2, float* dsncff2, float* dsncff4, float* dsncff6, float* jnet) {
    int lsfclr = ls * LR;

    int lkl = lklr[lsfclr + LEFT];
    int lkr = lklr[lsfclr + RIGHT];

    if (lkl < 0) {
        int idirr = idirlr[lsfclr + RIGHT];
        sanm2n_calculateJnet1n(ls, RIGHT, ng, ng2, nsurf, lklr, idirlr, sgnlr, hmesh, albedo(LEFT, idirr), xsadf, m251, m253, diagD, matM, flux, trlcff1, eta1, eta2, dsncff2, dsncff4, dsncff6, jnet);
    }
    else if (lkr < 0) {
        int idirr = idirlr[lsfclr + LEFT];
        sanm2n_calculateJnet1n(ls, LEFT, ng, ng2, nsurf, lklr, idirlr, sgnlr, hmesh, albedo(RIGHT, idirr), xsadf, m251, m253, diagD, matM, flux, trlcff1, eta1, eta2, dsncff2, dsncff4, dsncff6, jnet);
    }
    else {
        sanm2n_calculateJnet2n(ls, ng, ng2, nsurf, lklr, idirlr, sgnlr, hmesh, xsadf, m260, m262, m264, diagD, diagDI, matM, matMI, flux, trlcff0, trlcff1, trlcff2, mu, tau, eta1, eta2, dsncff2, dsncff4, dsncff6, jnet);
    }


}

__host__ __device__ void sanm2n_calculateJnet2n(const int& ls, const int& ng, const int& ng2, int& nsurf, int* lklr, int* idirlr, int* sgnlr, float* hmesh, XS_PRECISION* xsadf, float* m260, float* m262, float* m264, float* diagD, float* diagDI, float* matM, float* matMI, double* flux, float* trlcff0, float* trlcff1, float* trlcff2, float* mu, float* tau, float* eta1, float* eta2, float* dsncff2, float* dsncff4, float* dsncff6, float* jnet)
{
    int lsfclr = ls * LR;

    int lkl = lklr[lsfclr + LEFT];
    int lkr = lklr[lsfclr + RIGHT];

    int idirl = idirlr[lsfclr + LEFT];
    int idirr = idirlr[lsfclr + RIGHT];
    int sgnl = sgnlr[lsfclr + LEFT];
    int sgnr = sgnlr[lsfclr + RIGHT];
    int lkdl = lkl * NDIRMAX + idirl;
    int lkdr = lkr * NDIRMAX + idirr;

    float adf[2][LR], diagDj[2][LR], tempz[2][2], tempzI[2][2], zeta1[2][2], zeta2[2], bfc[2], mat1g[2][2];

    for (size_t ig = 0; ig < ng; ig++)
    {
        adf[ig][LEFT] = xsadf(ig, lkl);
        adf[ig][RIGHT] = xsadf(ig, lkr);
        diagDj[ig][LEFT] = 0.5 * hmesh(idirl, lkl) * diagD(ig, lkdl);
        diagDj[ig][RIGHT] = 0.5 * hmesh(idirr, lkr) * diagD(ig, lkdr);
    }

    //zeta1 = (mur + I + taur)_inv * (mul + I + taul)
    tempz[0][0] = (mu(0, 0, lkdr) + tau(0, 0, lkdr) + 1) * adf[0][RIGHT];
    tempz[1][0] = (mu(1, 0, lkdr) + tau(1, 0, lkdr)) * adf[0][RIGHT];
    tempz[0][1] = (mu(0, 1, lkdr) + tau(0, 1, lkdr)) * adf[1][RIGHT];
    tempz[1][1] = (mu(1, 1, lkdr) + tau(1, 1, lkdr) + 1) * adf[1][RIGHT];

    auto rdet = 1 / (tempz[0][0] * tempz[1][1] - tempz[1][0] * tempz[0][1]);
    tempzI[0][0] = rdet * tempz[1][1];
    tempzI[1][0] = -rdet * tempz[1][0];
    tempzI[0][1] = -rdet * tempz[0][1];
    tempzI[1][1] = rdet * tempz[0][0];

    tempz[0][0] = (mu(0, 0, lkdl) + tau(0, 0, lkdl) + 1) * adf[0][LEFT];
    tempz[1][0] = (mu(1, 0, lkdl) + tau(1, 0, lkdl)) * adf[0][LEFT];
    tempz[0][1] = (mu(0, 1, lkdl) + tau(0, 1, lkdl)) * adf[1][LEFT];
    tempz[1][1] = (mu(1, 1, lkdl) + tau(1, 1, lkdl) + 1) * adf[1][LEFT];

    zeta1[0][0] = tempzI[0][0] * tempz[0][0] + tempzI[1][0] * tempz[0][1];
    zeta1[1][0] = tempzI[0][0] * tempz[1][0] + tempzI[1][0] * tempz[1][1];
    zeta1[0][1] = tempzI[0][1] * tempz[0][0] + tempzI[1][1] * tempz[0][1];
    zeta1[1][1] = tempzI[0][1] * tempz[1][0] + tempzI[1][1] * tempz[1][1];

    for (size_t ig = 0; ig < ng; ig++)
    {
        bfc[ig] = adf[ig][RIGHT] * (dsncff2(ig, lkdr) + dsncff4(ig, lkdr) + dsncff6(ig, lkdr)
                                    + flux(ig, lkr) + matMI(0, ig, lkr) * sgnr * trlcff1(0, lkdr)
                                    + matMI(1, ig, lkr) * sgnr * trlcff1(1, lkdr))
                  + adf[ig][LEFT] * (-dsncff2(ig, lkdl) - dsncff4(ig, lkdl) - dsncff6(ig, lkdl)
                                     - flux(ig, lkl) + matMI(0, ig, lkl) * sgnl * trlcff1(0, lkdl)
                                     + matMI(1, ig, lkl) * sgnl * trlcff1(1, lkdl));
    }

    for (size_t ig = 0; ig < ng; ig++)
    {
        zeta2[ig] = tempzI[0][ig] * bfc[0] + tempzI[1][ig] * bfc[1];
    }

    //tempz = mur + 6 * I + eta1 * taur
    tempz[0][0] = diagDj[0][RIGHT] * (mu(0, 0, lkdr) + 6 + eta1(0, lkdr) * tau(0, 0, lkdr));
    tempz[1][0] = diagDj[0][RIGHT] * (mu(1, 0, lkdr) + eta1(0, lkdr) * tau(1, 0, lkdr));
    tempz[0][1] = diagDj[1][RIGHT] * (mu(0, 1, lkdr) + eta1(1, lkdr) * tau(0, 1, lkdr));
    tempz[1][1] = diagDj[1][RIGHT] * (mu(1, 1, lkdr) + 6 + eta1(1, lkdr) * tau(1, 1, lkdr));


    //mat1g = mul + 6 * I + eta1 * taul - tempzI
    mat1g[0][0] = -diagDj[0][LEFT] * (mu(0, 0, lkdl) + 6 + eta1(0, lkdl) * tau(0, 0, lkdl)) - tempz[0][0] * zeta1[0][0] - tempz[1][0] * zeta1[0][1];
    mat1g[1][0] = -diagDj[0][LEFT] * (mu(1, 0, lkdl) + eta1(0, lkdl) * tau(1, 0, lkdl)) - tempz[0][0] * zeta1[1][0] - tempz[1][0] * zeta1[1][1];
    mat1g[0][1] = -diagDj[1][LEFT] * (mu(0, 1, lkdl) + eta1(1, lkdl) * tau(0, 1, lkdl)) - tempz[0][1] * zeta1[0][0] - tempz[1][1] * zeta1[0][1];
    mat1g[1][1] = -diagDj[1][LEFT] * (mu(1, 1, lkdl) + 6 + eta1(1, lkdl) * tau(1, 1, lkdl)) - tempz[0][1] * zeta1[1][0] - tempz[1][1] * zeta1[1][1];


    float bcc[2], vec1g[2];

    for (size_t ig = 0; ig < ng; ig++)
    {
        bcc[ig] = diagDj[ig][LEFT] * (3 * dsncff2(ig, lkdl) + 10 * dsncff4(ig, lkdl) + eta2(ig, lkdl) * dsncff6(ig, lkdl))
                  + diagDj[ig][RIGHT] * (3 * dsncff2(ig, lkdr) + 10 * dsncff4(ig, lkdr) + eta2(ig, lkdr) * dsncff6(ig, lkdr));
        vec1g[ig] = bcc[ig]
                    - diagDj[ig][LEFT] * (matMI(0, ig, lkl) * sgnl * trlcff1(0, lkdl) + matMI(0, ig, lkl) * sgnl * trlcff1(0, lkdl))
                    + diagDj[ig][RIGHT] * (matMI(1, ig, lkr) * sgnr * trlcff1(1, lkdr) + matMI(1, ig, lkr) * sgnr * trlcff1(1, lkdr))
                    - (tempz[0][ig] * zeta2[0] + tempz[1][ig] * zeta2[1]);

    }

    rdet = 1 / (mat1g[0][0] * mat1g[1][1] - mat1g[1][0] * mat1g[0][1]);
    auto tmp = mat1g[0][0];
    mat1g[0][0] = rdet * mat1g[1][1];
    mat1g[1][0] = -rdet * mat1g[1][0];
    mat1g[0][1] = -rdet * mat1g[0][1];
    mat1g[1][1] = rdet * tmp;

    float oddcff[3][2];

    oddcff[1][0] = zeta2[0] - (zeta1[0][0] * (mat1g[0][0] * vec1g[0] + mat1g[1][0] * vec1g[1])
                               + zeta1[1][0] * (mat1g[0][1] * vec1g[0] + mat1g[1][1] * vec1g[1]));
    oddcff[1][1] = zeta2[1] - (zeta1[0][1] * (mat1g[0][0] * vec1g[0] + mat1g[1][0] * vec1g[1])
                               + zeta1[1][1] * (mat1g[0][1] * vec1g[0] + mat1g[1][1] * vec1g[1]));

    oddcff[2][0] = tau(0, 0, lkdr) * oddcff[1][0] + tau(1, 0, lkdr) * oddcff[1][1];
    oddcff[2][1] = tau(0, 1, lkdr) * oddcff[1][0] + tau(1, 1, lkdr) * oddcff[1][1];

    oddcff[0][0] = mu(0, 0, lkdr) * oddcff[1][0] - matMI(0, 0, lkr) * sgnr * trlcff1(0, lkdr)
                   + mu(1, 0, lkdr) * oddcff[1][1] - matMI(1, 0, lkr) * sgnr * trlcff1(1, lkdr);
    oddcff[0][1] = mu(0, 1, lkdr) * oddcff[1][0] - matMI(0, 1, lkr) * sgnr * trlcff1(0, lkdr)
                   + mu(1, 1, lkdr) * oddcff[1][1] - matMI(1, 1, lkr) * sgnr * trlcff1(1, lkdr);

    for (size_t ig = 0; ig < ng; ig++)
    {
        jnet(ig, ls) = sgnr * hmesh(idirr, lkr) * 0.5 * diagD(ig, lkdr) * (
                -1.0 * oddcff[0][ig] + 3 * dsncff2(ig, lkdr) - 6 * oddcff[1][ig] + 10 * dsncff4(ig, lkdr)
                - eta1(ig, lkdr) * oddcff[2][ig] + eta2(ig, lkdr) * dsncff6(ig, lkdr));
    }

}

__host__ __device__ void sanm2n_calculateJnet1n(const int& ls, const int& lr, const int& ng, const int& ng2, int& nsurf, int* lklr, int* idirlr, int* sgnlr, float* hmesh, const float& alb, XS_PRECISION* xsadf, float* m251, float* m253, float* diagD, float* matM, double* flux, float* trlcff1, float* eta1, float* eta2, float* dsncff2, float* dsncff4, float* dsncff6, float* jnet)
{
    int lsfclr = ls * LR;

    int lk = lklr[lsfclr + lr];
    int idir = idirlr[lsfclr + lr];
    int sgn = sgnlr[lsfclr + lr];
    int lkd = lk * NDIRMAX + idir;

    float mat3g[6][6], vec3g[6], cffodd[6], diagDj[2];

    float a11[2][2], a12[2], a13[2], a22[2][2], a23[2], a31[2], a32[2], a33[2];
    float b1[2], b2[2];

    //1, 1
    int irow = 0; int icol = 0;
    for (size_t ige = 0; ige < ng; ige++)
    {
        for (size_t igs = 0; igs < ng; igs++)
        {
            a11[igs][ige] = matM(igs, ige, lk) * m011;
            mat3g[ng * icol + igs][ng * irow + ige] = matM(igs, ige, lk) * m011;
        }
    }

    //1, 2
    icol = 1;
    for (size_t ig = 0; ig < ng; ig++)
    {
        a12[ig] = -diagD(ig, lkd) * m231;
        mat3g[ng * icol + ig][ng * irow + ig] = -diagD(ig, lkd) * m231;
    }

    //1, 3
    icol = 2;
    for (size_t ig = 0; ig < ng; ig++)
    {
        a13[ig] = -diagD(ig, lkd) * m251(ig, lkd);
        mat3g[ng * icol + ig][ng * irow + ig] = -diagD(ig, lkd) * m251(ig, lkd);
    }

    //2,2
    irow = 1; icol = 1;
    for (size_t ige = 0; ige < ng; ige++)
    {
        for (size_t igs = 0; igs < ng; igs++)
        {
            a22[igs][ige] = matM(igs, ige, lk) * m033;
            mat3g[ng * icol + igs][ng * irow + ige] = matM(igs, ige, lk) * m033;
        }
    }

    //2, 3
    icol = 2;
    for (size_t ig = 0; ig < ng; ig++)
    {
        a23[ig] = -diagD(ig, lkd) * m253(ig, lkd);
        mat3g[ng * icol + ig][ng * irow + ig] = -diagD(ig, lkd) * m253(ig, lkd);
    }

    for (size_t ig = 0; ig < ng; ig++)
    {
        diagDj[ig] = 0.5 * hmesh(idir, lk) * diagD(ig, lkd);
    }

    //3,1
    irow = 2;  icol = 0;
    for (size_t ig = 0; ig < ng; ig++)
    {
        a31[ig] = diagDj[ig] + alb;
        mat3g[ng * icol + ig][ng * irow + ig] = diagDj[ig] + alb;
    }

    //3,2
    irow = 2;  icol = 1;
    for (size_t ig = 0; ig < ng; ig++)
    {
        a32[ig] = 6 * diagDj[ig] + alb;
        mat3g[ng * icol + ig][ng * irow + ig] = 6 * diagDj[ig] + alb;
    }


    //3,3
    irow = 2;  icol = 2;
    for (size_t ig = 0; ig < ng; ig++)
    {
        a33[ig] = diagDj[ig] * eta1(ig, lkd) + alb;
        mat3g[ng * icol + ig][ng * irow + ig] = diagDj[ig] * eta1(ig, lkd) + alb;
    }


    //make right vector
    irow = 0;
    for (size_t ig = 0; ig < d_ng; ig++)
    {
        b1[ig] = -m011 * trlcff1(ig, lkd);
        vec3g[ng * irow + ig] = -m011 * trlcff1(ig, lkd);
    }
    irow = 2;

    for (size_t ig = 0; ig < d_ng; ig++)
    {
        b2[ig] = -sgn * (diagDj[ig] * (3 * dsncff2(ig, lkd) + 10 * dsncff4(ig, lkd) + eta2(ig, lkd) * dsncff6(ig, lkd)) + alb * (flux(ig, lk) + dsncff2(ig, lkd) + dsncff4(ig, lkd) + dsncff6(ig, lkd)));

        vec3g[ng * irow + ig] = -sgn * (diagDj[ig] * (3 * dsncff2(ig, lkd) + 10 * dsncff4(ig, lkd) + eta2(ig, lkd) * dsncff6(ig, lkd)) + alb * (flux(ig, lk) + dsncff2(ig, lkd) + dsncff4(ig, lkd) + dsncff6(ig, lkd)));
    }


    for (size_t ige = 0; ige < ng; ige++)
    {
        for (size_t igs = 0; igs < ng; igs++)
        {
            a22[igs][ige] = 1 / a23[ige] * a22[igs][ige];
        }
        a13[ige] = a13[ige] / a33[ige];
    }

    float a[2][2] = {0.0};
    for (size_t ige = 0; ige < ng; ige++)
    {
        a[ige][ige] = - a13[ige] * a31[ige];

        for (size_t igs = 0; igs < ng; igs++)
        {
            a[igs][ige] = a11[igs][ige]+a12[ige]*a22[igs][ige]+a13[ige]*a32[ige]*a22[igs][ige];
        }

        b1[ige] = b1[ige] - a13[ige] * b2[ige];
    }

    float oddcff[3][2];

    float rdet = 1 / (a[0][0] * a[1][1] - a[1][0] * a[0][1]);
    a11[0][0] = rdet * a[1][1];
    a11[1][0] = -rdet * a[1][0];
    a11[0][1] = -rdet * a[0][1];
    a11[1][1] = rdet * a[0][0];


    for (size_t ig = 0; ig < ng; ig++)
    {
        oddcff[0][ig] = a11[0][ig] * b1[0] + a11[1][ig] * b1[1];
    }

    for (size_t ig = 0; ig < ng; ig++)
    {
        oddcff[1][ig] = - a22[0][ig] * oddcff[0][0] - a11[1][ig] * oddcff[0][1];
    }

    for (size_t ig = 0; ig < ng; ig++)
    {
        oddcff[2][ig] = (b2[ig]-a31[ig]*oddcff[0][ig]-a32[ig]*oddcff[1][ig])/a33[ig];
    }

    for (size_t ig = 0; ig < ng; ig++)
    {
        jnet(ig, ls) = - hmesh(idir, lk) * 0.5 * diagD(ig, lkd) * (
                oddcff[0][ig] + 6 * oddcff[1][ig] + eta1(ig, lkd) * oddcff[1][ig]
                + sgn * (3 * dsncff2(ig, lkd) + 10 * dsncff4(ig, lkd) + eta2(ig, lkd) * dsncff6(ig, lkd)));
    }
}