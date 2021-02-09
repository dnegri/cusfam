//
// Created by JOO IL YOON on 2021/01/25.
//

#include "Feedback.h"

#define pow2(l2d, k)  pow[k*_g.nxy()+l2d]
#define pow3(l)  pow[l]
#define bu(l)  bu[l]


Feedback::Feedback(Geometry& g, SteamTable& steam) : _g(g), _steam(steam) {

}

Feedback::~Feedback()
{
}


void Feedback::initDelta(const float& ppm) {

    for (int l = 0; l < _g.nxyz(); ++l)
    {
        dppm(l) = ppm * dm(l)/dm0(l) - ppm0(l);
        dtf(l) = sqrt(tf(l)) - stf0(l);
        dtm(l) = tm(l) - tm0(l);
        ddm(l) = dm(l) - dm0(l);
    }
}
void Feedback::allocate() {

    _tf = new float[_g.nxyz()]{};
    _tm = new float[_g.nxyz()]{};
    _dm = new float[_g.nxyz()]{};
    _dppm = new float[_g.nxyz()]{};
    _dtf = new float[_g.nxyz()]{};
    _dtm = new float[_g.nxyz()]{};
    _ddm = new float[_g.nxyz()]{};
    _ppm0 = new float[_g.nxyz()]{};
    _stf0 = new float[_g.nxyz()]{};
    _tm0 = new float[_g.nxyz()]{};
    _dm0 = new float[_g.nxyz()]{};
    _chflow = new float[_g.nxy()]{};
    _fueltype = new int[_g.nxyz()];
    _frodn = new float[_g.nxy()];
}

void Feedback::initTFTable(const int& nft) {

    _nft = nft;
    _ntfbu = new int[nft];
    _ntfpow = new int[nft];
    _tfbu = new float[nft * TF_POINT];
    _tfpow = new float[nft * TF_POINT];
    _tftable = new float[nft * TF_POINT * TF_POINT];
}

void Feedback::updateTf(const int& l, const float* pow, const float* bu) {

    // powlin     : integrated nodal _power in w
    // qprime     : node average linear _power density in w/cm

    int ift = fueltype(l);
    if (ift < 0) return;

    float qprime = pow3(l) / _g.hmesh(ZDIR, l) * _heatfrac;
    qprime = qprime / frodn(l % _g.nxy());

    int i = 1;
    for (; i < ntfbu(ift) - 1; ++i) {
        if (bu(l) < tfbu(i, ift)) break;
    }

    int ib[2];
    float bus[2];

    ib[0] = i - 1;
    ib[1] = i;
    bus[0] = tfbu(ib[0], ift);
    bus[1] = tfbu(ib[1], ift);

    i = 1;
    for (; i < ntfpow(ift) - 2; ++i) {
        if (qprime < tfpow(i, ift)) break;
    }

    int ip[3];
    float pws[3];
    ip[0] = i - 1;
    ip[1] = i;
    ip[2] = i + 1;
    pws[0] = tfpow(ip[0], ift);
    pws[1] = tfpow(ip[1], ift);
    pws[2] = tfpow(ip[2], ift);

    float rx12 = 1. / (bus[1] - bus[0]);

    float yp[3];
    for (int idx = 0; idx < 3; ++idx) {
        float y1 = tftable(ib[0], ip[idx], ift);
        float y2 = tftable(ib[1], ip[idx], ift);
        yp[idx] = (y2 - y1) * rx12 * (bu(l) - bus[0]) + y1;
    }

    float a1 = (qprime - pws[1]) * (qprime - pws[2]) / ((pws[0] - pws[1]) * (pws[0] - pws[2]));
    float a2 = (qprime - pws[0]) * (qprime - pws[2]) / ((pws[1] - pws[0]) * (pws[1] - pws[2]));
    float a3 = (qprime - pws[0]) * (qprime - pws[1]) / ((pws[2] - pws[0]) * (pws[2] - pws[1]));

    float deltf = a1 * yp[0] + a2 * yp[1] + a3 * yp[2];

    tf(l) = tm(l) + deltf + KELVIN;
    dtf(l) = sqrt(tf(l)) - stf0(l);
}

void Feedback::updateTm(const int& l2d, const float* pow, int& nboiling) {

    float hlow = _hin;

    if (chflow(l2d) != 0.0) {
        int l = l2d;
        for (int k = 0; k < _g.nz(); ++k) {
            float hup = hlow + pow2(l2d, k) / chflow(l2d);
            float havg = (hlow + hup) * 0.5;

            SteamError err;
            _steam.checkEnthalpy(havg, err);
            if (err == STEAM_TABLE_ERROR_MAXENTH) {
                nboiling = nboiling + 1;
                _steam.getSatTemperature(tm(l2d, k));
                _steam.getDensity(havg, dm(l2d, k));
            }
            else {
                _steam.getTemperature(havg, tm(l2d, k));
                _steam.getDensity(havg, dm(l2d, k));
            }

            dtm(l) = tm(l2d, k) - tm0(l);
            ddm(l) = dm(l2d, k) - dm0(l);

            hlow = hup;

            l = l2d + _g.nxy();
        }
    }
    else {
        int l = l2d;
        for (int k = 0; k < _g.nz(); ++k) {
            tm(l2d, k) = _tin;
            dm(l2d, k) = _din;
            dtm(l) = tm(l2d, k) - tm0(l);
            ddm(l) = dm(l2d, k) - dm0(l);
            l = l2d + _g.nxy();
        }
    }
}

void Feedback::updatePressure(const float& press) {
    _steam.setPressure(press);
}

__host__ __device__ void Feedback::updateTin(const float& tin)
{
    _tin = tin;
    _steam.getEnthalpy(tin, _hin);
    _steam.getDensity(tin, _din);
}

__host__ __device__ void Feedback::updateTf(const float* power, const float* burnup)
{
#pragma omp parallel for
    for (size_t l = 0; l < _g.nxyz(); l++)
    {
        updateTf(l, power, burnup);
    }
}

__host__ __device__ void Feedback::updateTm(const float* power, int& nboiling)
{
    nboiling = 0;
#pragma omp parallel for
    for (size_t l2d = 0; l2d < _g.nxy(); l2d++)
    {
        updateTm(l2d, power, nboiling);
    }
}

void Feedback::updatePPM(const float& ppm) {
#pragma omp parallel for
    for (int l = 0; l < _g.nxyz(); ++l) {
        dppm(l) = ppm * dm(l)/dm0(l) - ppm0(l);
    }
}
