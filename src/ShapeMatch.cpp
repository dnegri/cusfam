

//
// Created by JOO IL YOON on 2022/06/19.
//
#include "ShapeMatch.h"
#include "JIArray.h"
#include "BICGCMFD.h"

ShapeMatch::ShapeMatch(Geometry& g, CrossSection& xs) : g(g), xs(xs) {

    float2 albedo(LR, NDIRMAX);
    albedo(LEFT, ZDIR) = g.albedo(LEFT, ZDIR);
    albedo(RIGHT, ZDIR) = g.albedo(LEFT, ZDIR);

    int nsurf = 4 * g.nz() + g.nz() + 1;
    g1d.setBoundaryCondition(0, 360, albedo.data());
    g1d.initDimension(g.ng(), 1, g.nz(), 1, 1, nsurf, 1);

    int nxs[1], nxe[1], nys[1], nye[1], ijtol[1], neibr[4]{};
    float2 hmesh(3, g.nz());
    nxs[0] = 1;
    nxe[0] = 1;
    nys[0] = 1;
    nye[0] = 1;
    ijtol[0] = 1;
    for (int k = 0; k < g.nz(); ++k) {
        hmesh(XDIR, k) = 1.0;
        hmesh(YDIR, k) = 1.0;
        hmesh(ZDIR, k) = g.hz(k);
    }

    int nxyfa = 1;
    int ncellfa = 1;
    int latol[1];
    int larot[1];
    latol[0] = 1;
    larot[0] = 0;
    int ncorn = 4;
    int lctol[4], ltolc[4];
    for (int lc = 0; lc < 4; ++lc) {
        lctol[lc] = 1;
        ltolc[lc] = lc + 1;
    }

    g1d.initIndex(nxs, nxe, nys, nye, ijtol, neibr, hmesh.data());
    g1d.initAssemblyIndex(nxyfa, ncellfa, latol, larot);
    g1d.initCorner(ncorn, lctol, ltolc);

    xs1d.init(g.ng(), 1, g.nz(), g.nz() - 1);

    alloc(dxsa, g.ng(), g.nz());
    alloc(xmacp, g.ng(), g.nz());
    alloc(pow1d, g.nz());
    alloc(phi1d, g.ng(), g.nz());
    alloc(lkg1d, g.ng(), g.nz());
    alloc(xstf1d0, g.ng(), g.nz());
    alloc(diffa, g.ng(), g.nz());
    alloc(jnet1d, g.ng(), g.nsurf());
    alloc(phis1d, g.ng(), g.nsurf());

    alloc(crntol, g.ng(), g.nz());
    alloc(crntor, g.ng(), g.nz());
    alloc(crntil, g.ng(), g.nz());
    alloc(crntir, g.ng(), g.nz());

    alloc(c1gu, g.ng(), g.nz());
    alloc(c2gu, g.ng(), g.nz());
    alloc(c3gu, g.ng(), g.nz());
    alloc(c4gu, g.ng(), g.nz());
    alloc(c5gu, g.ng(), g.nz());
    alloc(c6gu, g.ng(), g.nz());
    alloc(a1gu, g.ng(), g.nz());
    alloc(a2gu, g.ng(), g.nz());
    alloc(a3gu, g.ng(), g.nz());
    alloc(a4gu, g.ng(), g.nz());


    alloc(vol1d, g.nz());

    int lk = -1;
    for (int k = 0; k < g.nz(); ++k) {
        for (int l = 0; l < g.nxy(); ++l) {
            ++lk;
            vol1d(k) += g.vol(lk);
        }
    }

    //FIXME average node-wise CHI in updateAxialData
    for (int k = 0; k < g.nz(); ++k) {
        xs1d.chif(0, k) = 1.0;
        for (int ig = 1; ig < g.ng(); ++ig) {
            xs1d.chif(ig, k) = 0.0;
        }
    }

    cmfd1d = new BICGCMFD(g1d, xs1d);
    cmfd1d->init();
    cmfd1d->setNcmfd(5);
    cmfd1d->setEshift(0.04);
    maxiter = 100;


}

ShapeMatch::~ShapeMatch() {

}

void ShapeMatch::updateAxialData(const double& eigv, const double2& flux, const double2& jnet, const double1& powshp) {

    phi1d = 0.0;
    std::fill(xs1d.xsnf(0), xs1d.xsnf(0)+g1d.ng()*g1d.nz(), 0.0);
    std::fill(xs1d.xsdf(0), xs1d.xsdf(0)+g1d.ng()*g1d.nz(), 0.0);
    std::fill(xs1d.xstf(0), xs1d.xstf(0)+g1d.ng()*g1d.nz(), 0.0);
    std::fill(xs1d.xskf(0), xs1d.xskf(0)+g1d.ng()*g1d.nz(), 0.0);
    std::fill(xs1d.xssf(0), xs1d.xssf(0)+g1d.ng()*g1d.nz(), 0.0);

    lkg1d = 0.0;
    xmacp = 0.0;
    pow1d = 0.0;

    double reigv = 1. / eigv;

    for (int k = 0; k < g.nz(); ++k) {
        int lk0 = k * g.nxy();
        for (int l = 0; l < g.nxy(); ++l) {
            auto lk = lk0 + l;
            for (int ig = 0; ig < g.ng(); ig++) {
                phi1d(ig, k) += flux(ig, lk) * g.vol(lk);
                xmacp(ig, k) += reigv*xs.xsnf(ig, lk) * flux(ig, lk) * g.vol(lk);
                xs1d.xsnf(ig, k) += xs.xsnf(ig, lk) * flux(ig, lk) * g.vol(lk);
                xs1d.xsdf(ig, k) += xs.xsdf(ig, lk) * flux(ig, lk) * g.vol(lk);
                xs1d.xstf(ig, k) += xs.xstf(ig, lk) * flux(ig, lk) * g.vol(lk);
                xs1d.xskf(ig, k) += xs.xskf(ig, lk) * flux(ig, lk) * g.vol(lk);
                for (int igs = 0; igs < g.ng(); igs++) {
                    xs1d.xssf(igs, ig, k) += xs.xssf(igs, ig, lk) * flux(igs, lk) * g.vol(lk);
                }
            }
        }


//        double hxy = 0.0;
        for (int j = 0; j < g.ny(); ++j) {
            auto ib = g.nxs(j);
            auto ie = g.nxe(j) - 1;

            auto lkb = g.ijtol(ib, j) + lk0;
            auto lke = g.ijtol(ie, j) + lk0;

            auto lsb = g.lktosfc(LEFT, XDIR, lkb);
            auto lse = g.lktosfc(RIGHT, XDIR, lke);

//            hxy += g.hmesh(YDIR, lkb)*g.hmesh(ZDIR, lkb);
            for (int ig = 0; ig < g.ng(); ig++) {
                lkg1d(ig, k) += (jnet(ig, lse) - jnet(ig, lsb)) * g.hmesh(YDIR, lkb) * g.hmesh(ZDIR, lkb);
            }
        }

        for (int i = 0; i < g.nx(); ++i) {
            auto jb = g.nys(i);
            auto je = g.nye(i) - 1;

            auto lkb = g.ijtol(i, jb) + lk0;
            auto lke = g.ijtol(i, je) + lk0;

            auto lsb = g.lktosfc(LEFT, YDIR, lkb);
            auto lse = g.lktosfc(RIGHT, YDIR, lke);

//            hxy += g.hmesh(XDIR, lkb)*g.hmesh(ZDIR, lkb);

            for (int ig = 0; ig < g.ng(); ig++) {
                lkg1d(ig, k) += (jnet(ig, lse) - jnet(ig, lsb)) * g.hmesh(XDIR, lkb) * g.hmesh(ZDIR, lkb);
            }
        }

//        for (int l = 0; l < g.nxy(); ++l) {
//            auto lk = lk0 + l;
//            auto lsb = g.lktosfc(LEFT, ZDIR, lk);
//            auto lse = g.lktosfc(RIGHT, ZDIR, lk);
//            for (int ig = 0; ig < g.ng(); ig++) {
//                lkg1d(ig, k) += (jnet(ig, lse) - jnet(ig, lsb)) * g.hmesh(XDIR, lk) * g.hmesh(YDIR, lk);
//            }
//        }
//        for (int ig = 0; ig < g.ng(); ig++) {
//            lkg1d(ig, k) = lkg1d(ig, k) / (phi1d(ig, k));
//        }

        for (int ig = 0; ig < g.ng(); ig++) {
            double rphivol = 1. / phi1d(ig, k);
            xmacp(ig, k) = xmacp(ig, k) * rphivol;
            xs1d.xsnf(ig, k) = xs1d.xsnf(ig, k) * rphivol;
            xs1d.xsdf(ig, k) = xs1d.xsdf(ig, k) * rphivol;
            xs1d.xstf(ig, k) = xs1d.xstf(ig, k) * rphivol;
            xs1d.xskf(ig, k) = xs1d.xskf(ig, k) * rphivol;
            for (int ige = 0; ige < g.ng(); ige++) {
                xs1d.xssf(ig, ige, k) = xs1d.xssf(ig, ige, k) * rphivol;
            }


            lkg1d(ig, k) = lkg1d(ig, k) * rphivol;
            xs1d.xstf(ig, k) = xs1d.xstf(ig, k) + lkg1d(ig, k);
            xstf1d0(ig, k) = xs1d.xstf(ig, k);
            diffa(ig, k) = xs1d.xsdf(ig, k) / g1d.hz(k);
            phi1d(ig, k) /= vol1d(k);

        }

    }

    for (int k = g1d.kbc(); k < g1d.kec(); ++k) {
        phi1d(0,k) = powshp(k)/(xs1d.xskf(0,k) + 0.25 * xs1d.xskf(1,k));
        phi1d(0,k) = phi1d(0,k) / g1d.hz(k);
        phi1d(1,k) = 0.25 * phi1d (0,k);
    }

    for (int k = 0; k < g1d.kbc(); ++k) {
        for (int ig = 0; ig < g1d.ng(); ++ig) {
            phi1d(ig,k) =phi1d(ig,g1d.kbc());
        }
    }

    for (int k = g1d.kec(); k < g1d.nz(); ++k) {
        for (int ig = 0; ig < g1d.ng(); ++ig) {
            phi1d(ig,k) =phi1d(ig,g1d.kec()-1);
        }
    }
    for (int k = 0; k < g1d.nz(); ++k) {
        for (int ig = 0; ig < g1d.ng(); ++ig) {
            pow1d(k) += phi1d(ig, k) * xs1d.xskf(ig, k);

            crntol(ig, k) = phi1d(ig, k) / 4.0;
            crntor(ig, k) = phi1d(ig, k) / 4.0;
            crntil(ig, k) = 0.0;
            crntir(ig, k) = 0.0;
        }
    }


#define SHAPE_MATCH_NODAL_BALANCE
#ifdef SHAPE_MATCH_NODAL_BALANCE
    for (int lk = 0; lk < g.nxyz(); ++lk) {
        double psi = 0.0;
        for (int ig = 0; ig < g.ng(); ++ig) {
            psi += xs.xsnf(ig, lk) * flux(ig, lk);
        }

        for (int ig = 0; ig < g.ng(); ++ig) {
            double sct = 0.0;
            for (int igs = 0; igs < g.ng(); ++igs) {
                sct += xs.xssf(igs, ig, lk) * flux(igs, lk);
            }

            double rem = xs.xstf(ig, lk) * flux(ig, lk);

            double lkg = 0.0;
            for (int idir = 0; idir < NDIRMAX; ++idir) {
                int lsl = g.lktosfc(LEFT, idir, lk);
                int lsr = g.lktosfc(RIGHT, idir, lk);
                lkg += (jnet(ig, lsr) - jnet(ig, lsl)) / g.hmesh(idir, lk);
            }

            double balance = abs(reigv * xs.chif(ig, lk) * psi + sct - rem - lkg) / rem;

//            assert(balance<1.0E-4);

//            cout << balance;


        }

    }


    for (int k = 0; k < g.nz(); ++k) {
        double psi = 0.0;
        for (int ig = 0; ig < g.ng(); ++ig) {
            psi += xs1d.xsnf(ig, k) * phi1d(ig, k);
        }


        for (int ig = 0; ig < g.ng(); ++ig) {
            double sct = 0.0;
            for (int igs = 0; igs < g.ng(); ++igs) {
                sct += xs1d.xssf(igs, ig, k) * phi1d(igs, k);
            }

            double rem = xs1d.xstf(ig, k) * phi1d(ig, k);
            double lkg = lkg1d(ig, k) * phi1d(ig, k);

            double balance = abs((xs1d.chif(ig, k) * psi + sct - rem) * vol1d(k) - lkg) / (rem * vol1d(k));

//            cout << balance;

//            assert(balance<1.0E-4);

        }

    }
#endif

}

double ShapeMatch::totfis() {
    double psi = 0.0;
    for (int k = 0; k < g1d.nz(); ++k) {
        for (int ig = 0; ig < g1d.ng(); ++ig) {
            psi += xs1d.xsnf(ig, k) * phi1d(ig, k) * g1d.hz(k);
        }
    }
    return psi;

}

void ShapeMatch::neminit() {
    for (int k = 0; k < g1d.nz(); ++k) {
        for (int ig = 0; ig < g1d.ng(); ++ig) {
            c1gu(ig, k) = 6.0 * diffa(ig, k) / (1. + 12. * diffa(ig, k));
            c3gu(ig, k) = -4.0 * c1gu(ig, k) / (3. + 12. * diffa(ig, k));
            c2gu(ig, k) = 1.0 - 4. * c1gu(ig, k) - c3gu(ig, k);
            c4gu(ig, k) = c1gu(ig, k) - 6. * diffa(ig, k) * c3gu(ig, k);
            c5gu(ig, k) = 2.0 * c1gu(ig, k) / g1d.hz(k);
        }
    }
}

void ShapeMatch::moment() {
    double c5d3 = 1.6666666667;
    double c7d3 = 2.333333333;

    double2 g1m(g1d.ng(), g1d.nz());
    double2 g2m(g1d.ng(), g1d.nz());
    double2 rhs1m(g1d.ng(), g1d.nz());
    double2 rhs2m(g1d.ng(), g1d.nz());

    for (int k = 0; k < g1d.nz(); k++) {
        for (int ig = 0; ig < g1d.ng(); ig++) {
//     ---- calculate coefficients for iteration
            auto dz = diffa(ig, k) / g1d.hz(k);
            g1m(ig, k) = 60. * dz + xs1d.xstf(ig, k);
            g2m(ig, k) = 140. * dz + xs1d.xstf(ig, k);
        }
    }
    for (int k = 0; k < g1d.nz(); k++) {
//     right hand side of momenstum equation
        rhs1m(0, k) = a1gu(0, k) * xmacp(0, k) + a1gu(1, k) * xmacp(1, k) - a1gu(0, k) * xs1d.xstf(0, k);
        rhs2m(0, k) = -a2gu(0, k) * xmacp(0, k) - a2gu(1, k) * xmacp(1, k) + a2gu(0, k) * xs1d.xstf(0, k);
        rhs1m(0, k) = c5d3 * rhs1m(0, k);
        rhs2m(0, k) = c7d3 * rhs2m(0, k);
        rhs1m(1, k) = a1gu(0, k) * xs1d.xssf(0, 1, k) - a1gu(1, k) * xs1d.xstf(1, k);
        rhs2m(1, k) = a2gu(1, k) * xs1d.xstf(1, k) - a2gu(0, k) * xs1d.xssf(0, 1, k);
        rhs1m(1, k) = c5d3 * rhs1m(1, k);
        rhs2m(1, k) = c7d3 * rhs2m(1, k);
    }
//        -----------------------------------------------------
//           solve moments equation
//        -----------------------------------------------------
//     ---- solve 2x2 matrices for a3gu and a4gu
    for (int k = 0; k < g1d.nz(); k++) {
        auto scr1 = g1m(0, k) - xmacp(0, k);
        auto scr2 = g2m(0, k) - xmacp(0, k);
        auto scr = xmacp(1, k) * xs1d.xssf(0, 1, k);
        auto det = 1.0 / (scr1 * g1m(1, k) - scr);
        a3gu(0, k) = det * (rhs1m(0, k) * g1m(1, k) + rhs1m(1, k) * xmacp(1, k));
        a3gu(1, k) = det * (rhs1m(0, k) * xs1d.xssf(0, 1, k) + rhs1m(1, k) * scr1);
        det = 1.0 / (scr2 * g2m(1, k) - scr);
        a4gu(0, k) = det * (rhs2m(0, k) * g2m(1, k) + rhs2m(1, k) * xmacp(1, k));
        a4gu(1, k) = det * (rhs2m(0, k) * xs1d.xssf(0, 1, k) + rhs2m(1, k) * scr2);
    }
}

void ShapeMatch::nemcoef() {
    for (int k = 0; k < g1d.nz(); k++) {

        if (k < g1d.kbc() || k >= g1d.kec()) {

            auto c1 = xs1d.xstf(0, k) / xs1d.xsdf(0, k);
            auto c3 = xs1d.xstf(1, k) / xs1d.xsdf(1, k);
            auto c4 = xs1d.xssf(0, 1, k) / xs1d.xsdf(1, k);

            auto kapp1 = sqrt(c1);
            auto kapp2 = sqrt(c3);
            auto beta = c4 / (c3 - c1);

            auto d1k1 = 2.0 * xs1d.xsdf(0, k) * kapp1;
            auto d2k1 = 2.0 * xs1d.xsdf(1, k) * kapp1;
            auto d2k2 = 2.0 * xs1d.xsdf(1, k) * kapp2;

            auto k1z = kapp1 * g1d.hz(k);
            auto k2z = kapp2 * g1d.hz(k);

            auto e0 = exp(k1z);
            auto e1 = 1.0 / e0;
            auto csh1 = 0.5 * (e0 + e1);
            auto snh1 = 0.5 * (e0 - e1);

            e0 = exp(k2z);
            e1 = 1.0 / e0;
            auto csh2 = 0.5 * (e0 + e1);
            auto snh2 = 0.5 * (e0 - e1);

            auto a11 = 1.0;
            auto a12 = -d1k1;
            auto a21 = csh1 + d1k1 * snh1;
            auto a22 = snh1 + d1k1 * csh1;

            auto deta = 4.0 / (a11 * a22 - a12 * a21);

            auto rk1 = deta * (a22 * crntil(0, k) - a12 * crntir(0, k));
            auto rk2 = deta * (a11 * crntir(0, k) - a21 * crntil(0, k));

            auto lhs1 = 4.0 * crntil(1, k) - beta * (rk1 - d2k1 * rk2);
            auto lhs2 = 4.0 * crntir(1, k) - beta * (csh1 + d2k1 * snh1) * rk1 - beta * (snh1 + d2k1 * csh1) * rk2;

            auto b11 = 1.0;
            auto b12 = -d2k2;
            auto b21 = csh2 + d2k2 * snh2;
            auto b22 = snh2 + d2k2 * csh2;

            auto detb = 1.0 / (b11 * b22 - b12 * b21);

            auto rk3 = detb * (b22 * lhs1 - b12 * lhs2);
            auto rk4 = detb * (b11 * lhs2 - b21 * lhs1);

            a1gu(0, k) = rk1;
            a2gu(0, k) = rk2;
            a3gu(0, k) = rk3;
            a4gu(0, k) = rk4;

            a1gu(1, k) = k1z;
            a2gu(1, k) = k2z;
            a3gu(1, k) = beta;

            c5gu(0, k) = snh1;
            c5gu(1, k) = snh2;
            c6gu(0, k) = csh1;
            c6gu(1, k) = csh2;
        }
    }
}

void ShapeMatch::thrmabs() {
    for (int k = g1d.kbc(); k < g1d.kec(); k++) {

        auto scr2 = 2. * (crntil(1, k) + crntir(1, k)) - a4gu(1, k);
        scr2 = scr2 * c5gu(1, k);
        auto sos2 = xs1d.xssf(0, 1, k) * phi1d(0, k);

        xs1d.xstf(1, k) = (scr2 + sos2) / phi1d(1, k) - c5gu(1, k);

        auto scr1 = 2. * (crntil(0, k) + crntir(0, k)) - a4gu(0, k);
        scr1 = scr1 * c5gu(0, k);
        auto sos1 = xmacp(0, k) * phi1d(0, k) + xmacp(1, k) * phi1d(1, k);

        xs1d.xstf(0, k) = (scr1 + sos1) / phi1d(0, k) - c5gu(0, k);

    }

}

double ShapeMatch::fluxcal(const double1& powshp) {
    double errflx = 0.0;
    for (int k = 0; k < g1d.nz(); k++) {

        auto phil1 = phi1d(0, k);
        auto phil2 = phi1d(1, k);

        if (k < g1d.kbc() || k >= g1d.kec()) {

            phi1d(0, k) = a1gu(0, k) * c5gu(0, k) + a2gu(0, k) * (c6gu(0, k) - 1.0);
            phi1d(0, k) = phi1d(0, k) / a1gu(1, k);

            phi1d(1, k) = a3gu(0, k) * c5gu(1, k) + a4gu(0, k) * (c6gu(1, k) - 1.0);
            phi1d(1, k) = phi1d(1, k) / a2gu(1, k);
            phi1d(1, k) = a3gu(1, k) * phi1d(0, k) + phi1d(1, k);

        } else {

            auto scr1 = 2. * (crntil(0, k) + crntir(0, k)) - a4gu(0, k);
            scr1 = scr1 * c5gu(0, k);
            auto sos1 = xmacp(0, k) * phi1d(0, k) + xmacp(1, k) * phi1d(1, k);

            phi1d(0, k) = (scr1 + sos1) / c6gu(0, k);

            auto scr2 = 2. * (crntil(1, k) + crntir(1, k)) - a4gu(1, k);
            scr2 = scr2 * c5gu(1, k);
            auto sos2 = xs1d.xssf(0, 1, k) * phi1d(0, k);

            phi1d(1, k) = (scr2 + sos2) / c6gu(1, k);

            auto p = xs1d.xskf(0, k) * phi1d(0, k) + xs1d.xskf(1, k) * phi1d(1, k);

            for (int ig = 0; ig < g1d.ng(); ++ig) {
                phi1d(ig, k) = phi1d(ig, k) * powshp(k) / (g1d.hz(k) * p);
            }
        }

        //           ---- calculate flux error

        auto err1 = abs(1. - phil1 / phi1d(0, k));
        auto err2 = abs(1. - phil2 / phi1d(1, k));
        auto err = max(err1, err2);
        if (err > errflx) {
            errflx = err;
        }
    }

    return errflx;
}

void ShapeMatch::currento() {

//     outgoing current calculation

    auto curmax = 0.0;
    auto errl1 = 0.0;
    auto errl2 = 0.0;
    auto errr1 = 0.0;
    auto errr2 = 0.0;
    for (int k = 0; k < g1d.nz(); k++) {
        if (k < g1d.kbc() || k >= g1d.kec()) {

            auto curl1 = crntol(0, k);
            auto curl2 = crntol(1, k);
            auto curr1 = crntor(0, k);
            auto curr2 = crntor(1, k);

            crntol(0, k) = 0.5 * a1gu(0, k);
            crntol(0, k) = crntol(0, k) - crntil(0, k);

            crntol(1, k) = a3gu(1, k) * a1gu(0, k) + a3gu(0, k);
            crntol(1, k) = 0.5 * crntol(1, k);
            crntol(1, k) = crntol(1, k) - crntil(1, k);

            auto a1 = a1gu(0, k) * c6gu(0, k) + a2gu(0, k) * c5gu(0, k);
            crntor(0, k) = 0.5 * a1;
            crntor(0, k) = crntor(0, k) - crntir(0, k);

            crntor(1, k) = a3gu(1, k) * a1 + a3gu(0, k) * c6gu(1, k) + a4gu(0, k) * c5gu(1, k);
            crntor(1, k) = 0.5 * crntor(1, k);
            crntor(1, k) = crntor(1, k) - crntir(1, k);

            if (crntol(0, k) != 0.0) errl1 = abs(1.0 - curl1 / crntol(0, k));
            if (crntol(1, k) != 0.0) errl2 = abs(1.0 - curl2 / crntol(1, k));
            if (crntor(0, k) != 0.0) errr1 = abs(1.0 - curr1 / crntor(0, k));
            if (crntor(1, k) != 0.0) errr2 = abs(1.0 - curr2 / crntor(1, k));
            auto errl = max(errl1, errl2);
            auto errr = max(errr1, errr2);
            auto err = max(errl, errr);
            if (err > curmax) curmax = err;
        } else {
            for (int ig = 0; ig < g1d.ng(); ig++) {

                auto curoldl = crntol(ig, k);
                auto curoldr = crntor(ig, k);
                auto scr1 = c1gu(ig, k) * (phi1d(ig, k) + a4gu(ig, k));

                crntol(ig, k) = scr1 + c2gu(ig, k) * crntil(ig, k)
                                + c3gu(ig, k) * crntir(ig, k)
                                - c4gu(ig, k) * a3gu(ig, k);

                crntor(ig, k) = scr1 + c3gu(ig, k) * crntil(ig, k)
                                + c2gu(ig, k) * crntir(ig, k)
                                + c4gu(ig, k) * a3gu(ig, k);

                auto errl = abs(1.0 - curoldl / crntol(ig, k));
                auto errr = abs(1.0 - curoldr / crntor(ig, k));
                auto err = max(errl, errr);
                if (err > curmax) curmax = err;
            }
        }
    }
//
}

void ShapeMatch::solve(const double& eigvt, const double2& flux, const double2& jnet, const double1& powshp) {


    updateAxialData(eigvt, flux, jnet, powshp);

    auto fisold = totfis();

    neminit();

    int mcy = 100;
    int icy = 0;
    for (icy = 0; icy < mcy; ++icy) {
        int it =0;
        for (it = 0; it < mcy; ++it) {
            for (int k = 0; k < g1d.nz(); ++k) {
                for (int ig = 0; ig < g1d.ng(); ++ig) {
                    c6gu(ig, k) = xs1d.xstf(ig, k) + c5gu(ig, k);
                }
            }
            // currenti
            for (int k = 0; k < g1d.nz(); ++k) {
                for (int ig = 0; ig < g1d.ng(); ++ig) {
                    if (k > 0) crntil(ig, k) = crntor(ig, k - 1);
                    if (k < g1d.nz() - 1) crntir(ig, k) = crntol(ig, k + 1);
                }
            }

            // fluxcoef
            for (int k = 0; k < g1d.nz(); ++k) {
                for (int ig = 0; ig < g1d.ng(); ++ig) {
                    auto phir = crntor(ig, k) + crntir(ig, k);
                    auto phil = crntol(ig, k) + crntil(ig, k);
                    a1gu(ig, k) = phir - phil;
                    a2gu(ig,k) = -phir - phil + phi1d(ig, k);
                }
            }

            moment();

            nemcoef();

            auto errflx = fluxcal(powshp);

            thrmabs();

            currento();

            if(errflx < 1.E-5) break;
        }

        auto fisnew = totfis();

        fisold = fisnew;

        auto eigrat = fisnew/fisold;

        auto erreig = abs(1 - eigrat);

        for (int k = g1d.kbc(); k < g1d.kec(); ++k) {
            for (int ig = 0; ig < g1d.ng(); ++ig) {
                xs1d.xstf(ig,k) = xs1d.xstf(ig,k) * eigrat;
            }
        }

        if(erreig < 1.E-6) break;
    }

    for (int k = g1d.kbc(); k < g1d.kec(); ++k) {
        for (int ig = 0; ig < g1d.ng(); ++ig) {
            dxsa(ig,k) = xs1d.xstf(ig,k) - xstf1d0(ig,k);
        }
    }

}
