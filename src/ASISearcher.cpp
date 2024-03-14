#include "ASISearcher.h"

ASISearcher::ASISearcher(Geometry &g_) : g(g_) {
    ndiv = 6;
    alloc(xediv, ndiv);
    alloc(xen1d, g.nz());
    alloc(xeratio, g.nz());
    alloc(p1dp, g.nz());
    alloc(p1d, g.nz());
    alloc0(fintg, 0, 6, -3, 4);
    alloc(coeff, 7);
    alloc(delmac, g.ng(), g.nxyz());

    xediv = 0.0;
    xen1d = 0.0;
    xeratio = 0.0;

    xeavg = 0.0;
    xemid = 0.0;
    xe23a = 0.0;
    xetmb = 0.0;
    delxe = 0.0;

    alloc(xeac, 2);
    alloc(xebc, 3);

    xeac  = {0.56, 5.80};
    xebc  = {2.50,-0.63,1.40};
    xextrp = 30.0;
    exthght = g.hzcore() + 2 * xextrp;

    calculateIntegralForXeDiv();
}

ASISearcher::~ASISearcher() {
}

void ASISearcher::reset(Depletion& d) {

    divideXenon(d);

    auto xedivsum = xediv.sum();
    xeavg = xedivsum / ndiv;
    xemid = (xediv(2) + xediv(3)) * 0.5 - xeavg;
    delxe = xeavg * 0.1;
    xe23a = (xediv(1) + xediv(2) + xediv(3) + xediv(4)) * 0.25 - xeavg;
    xetmb = ((xediv(4) + xediv(5)) - (xediv(0) + xediv(1))) * 0.5;

    ndavg = xedivsum / ndiv;
    ndcnt = (xediv(2) + xediv(3)) * 0.5;
    nddel = ndavg * 0.1;
}

void ASISearcher::search(const double &epsflx1, const int &nouter, Simon &simon, const SteadyOption &option, const double &targetASI, const double &epsASI) {

    double nddel0 = 0.0;
    double1 xef(g.nz());

    for (int i = 0; i < nouter; i++) {
        simon.runSteady(option);
        simon.generateResults();
        auto err = abs(simon.asi() - targetASI);

        if (err < epsASI)
            break;

        if (i == 0) {
            reset(simon.d());
            asi0 = simon.asi();
            asip = simon.asi();
            p1d = simon.pow1d();
            p1dp = simon.pow1d();
            if (simon.asi() > targetASI)
                delxe = -delxe;
            nddel0 = ((xediv(3) + xediv(4) + xediv(5)) - (xediv(0) + xediv(1) + xediv(2))) / 3;
        } else {
            auto tnddel = nddel;
            nddel = (targetASI - asi0) / (simon.asi() - asi0) * (nddel - nddel0) + nddel0;
            nddel0 = tnddel;
        }

        asi0 = simon.asi();

        generateXenonShape2(xef);
        changeXenonDensity(xef, simon.d(), simon.x());
    }
}

void ASISearcher::divideXenon(Depletion& d) {
    xediv = 0.0;
    xen1d = 0.0;
    auto xenao = 0.0;
    auto pos = 0.0;
    auto nodeSize = g.hzcore() / ndiv;
    auto limit = nodeSize;

    for (int k = g.kbc(); k < g.kec(); k++) {
        int lk0 = k * g.nxy();
        double volt = 0.0;
        for (int l = 0; l < g.nxy(); l++) {
            int lk = lk0 + l;
            if (d.dnst(XE45, lk) == 0.0)
                continue;

            xen1d(k) += d.dnst(XE45, lk) * g.vol(lk);
            volt += g.vol(lk);
        }
        xen1d(k) /= volt;
    }

    int idiv = 0;

    for (int k = g.kbc(); k < g.kec(); k++) {
        pos = pos + g.hz(k);
        if (pos >= limit) {
            xediv(idiv) = xediv(idiv) + xen1d(k) * (g.hz(k) - (pos - limit));
            idiv = idiv + 1;

            if (idiv == ndiv)
                break;

            xediv(idiv) = xediv(idiv) + xen1d(k) * (pos - limit);

            limit = limit + nodeSize;
        } else {
            xediv(idiv) = xediv(idiv) + xen1d(k) * g.hz(k);
        }
    }

    xediv /= nodeSize;
}

void ASISearcher::generateXenonShape2(double1 &xef) {

    double2 A(6, 6);
    double1 b(6);
    double1 fintg1(6);

    coeff(0) = ndavg;

    ndcnt2 = xeac(0) * (ndcnt - ndavg) + ndavg + xeac(1) * 1.e-11;
    ndtmb = 0.5 * xebc(0) * nddel + xebc(1) * ndavg * 1.e-2 + xebc(2) * 1.e-11;

    for (int i = 0; i < 6; i++) {
        A(i, 0) = 1.5 * (fintg(i, 1) - fintg(i, -1));                              // for ndcnt;
        A(i, 1) = 0.75 * (fintg(i, 2) - fintg(i, -2));                             // for ndcnt2;
        A(i, 2) = fintg(i, 3) + fintg(i, -3) - 2 * fintg(i, 0);                    // for nddel;
        A(i, 3) = 1.5 * (fintg(i, 3) + fintg(i, -3) - fintg(i, 1) - fintg(i, -1)); // for ndtmb;
        A(i, 4) = pow(-1, i + 1);
        A(i, 5) = 1;
    }

    b(0) = ndcnt - coeff(0);
    b(1) = ndcnt2 - coeff(0);
    b(2) = nddel;
    b(3) = ndtmb;
    b(4) = -coeff(0);
    b(5) = -coeff(0);
        
    double1 sol(coeff.getMemory(1), 6);
    invmatxvec1(A, b, sol, 6);

    xef = 0.0;
    double h = -1.0;

    getIntegral(h, fintg1);
    auto intgp = coeff(0) * h + (fintg1 * sol).sum();

    for (int k = g.kbc(); k < g.kec(); k++) {

        xef(k) = 0.0;
        h = h + g.hz(k) * 2 / g.hzcore();

        getIntegral(h, fintg1);
        auto intg = coeff(0) * h + (fintg1 * sol).sum();

        xef(k) = (intg - intgp) / (g.hz(k) * 2 / g.hzcore());

        intgp = intg;
    }
}

void ASISearcher::changeXenonDensity(const double1 &xef, Depletion& d, CrossSection& x) {

    for (int k = g.kbc(); k < g.kec(); k++) {
        xeratio(k) = xef(k) / xen1d(k) - 1;
        int lk0 = k * g.nxy();
        for (int l = 0; l < g.nxy(); l++) {
            int lk = lk0 + l;
            for (int ig = 0; ig < g.ng(); ig++) {
                x.delmac(ig, lk) = d.dnst(XE45, lk) * xeratio(k) * x.xsmica(ig, XE45, lk);
            }
        }
    }
}

void ASISearcher::calculateIntegralForXeDiv() {
    for (int i = -3; i <= 3; i++) {
        double x = i / 3.0;
        double1 a = fintg.slice<1>(i);
        getIntegral(x, a);
    }
}
void ASISearcher::getIntegral(const double &x, double1& fintg1) {
    fintg1(0) = 0.5 * (pow(x, 2));
    fintg1(1) = 0.5 * (pow(x, 3) - x);
    fintg1(2) = 0.125 * (5 * pow(x, 4) - 6 * pow(x, 2));
    fintg1(3) = 0.125 * (7 * pow(x, 5) - 10 * pow(x, 3) + 3 * x);
    fintg1(4) = 1. / 16. * (21 * pow(x, 6) - 35 * pow(x, 4) + 15 * pow(x, 2));
    fintg1(5) = 1. / 16. * (33 * pow(x, 7) - 63 * pow(x, 5) + 35 * pow(x, 3) - 5 * x);
}

void ASISearcher::invmatxvec1(double2 &mat, double1 &vec, double1 &sol, const int &n) {
    int1 pivot(n);
    double1 outvec(n);

    // initialize pivot
    for (int i = 0; i < n; i++) {
        pivot(i) = i;
    }

    // forward elimination
    for (int k = 0; k < n - 1; k++) {
        auto pmax = abs(mat(k, pivot(k)));
        auto ipmax = k;
        for (int i = k+1; i < n; i++) {
            if (abs(mat(k, pivot(i))) > pmax) {
                pmax = abs(mat(k, pivot(i)));
                ipmax = i;
            }
        }

        auto it = pivot(ipmax);
        pivot(ipmax) = pivot(k);
        pivot(k) = it;

        for (int i = k+1; i < n; i++) {
            auto fm = mat(k, pivot(i)) / mat(k, pivot(k));
            if (fm == 0)
                continue;
            for (int j = k; j < n; j++) {
                mat(j, pivot(i)) = mat(j, pivot(i)) - fm * mat(j, pivot(k));
            }
            vec(pivot(i)) = vec(pivot(i)) - fm * vec(pivot(k));
        }
    }

    // backward elimination
    outvec(pivot(n - 1)) = vec(pivot(n - 1)) / mat(n - 1, pivot(n - 1));
    for (int i = n - 2; i >= 0; i--) {
        double sum = 0;
        for (int j = i+1; j < n; j++) {
            sum = sum + mat(j, pivot(i)) * outvec(pivot(j));
        }
        outvec(pivot(i)) = (vec(pivot(i)) - sum) / mat(i, pivot(i));
    }

    for (int i = 0; i < n; i++) {
        sol(i) = outvec(pivot(i));
    }
}
