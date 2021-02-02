//
// Created by JOO IL YOON on 2021/01/25.
//

#include "Feedback.h"

#define pow(l2d, k)  pow[k*_g.nxy()+l2d]
#define pow(l)  pow[l]
#define bu(l)  bu[l]

Feedback::Feedback(Geometry& g, SteamTable& steam) : _g(g), _steam(steam) {
    _tf = new float[g.nxyz()]{};
    _tm = new float[g.nxyz()]{};
}

void Feedback::updateTf(const int& l, const float* pow, const float* bu) {

    // powlin     : integrated nodal power in w
    // qprime     : node average linear power density in w/cm
    float qprime = pow(l) / _g.hmesh(ZDIR, l) * _heatfrac;
    qprime = qprime / _g.npinbox();

    int i = 1;
    for (; i < _ntfbu - 1; ++i) {
        if (bu(l) < tfbu(i)) break;
    }

    int ib[2];
    float bus[2];

    ib[1] = i;
    ib[0] = i - 1;
    bus[0] = tfbu(ib[0]);
    bus[1] = tfbu(ib[1]);

    i = 1;
    for (; i < _ntfpow - 2; ++i) {
        if (qprime < tfpow(i)) break;
    }

    int ip[3];
    float pws[3];
    ip[0] = i - 1;
    ip[1] = i;
    ip[2] = i + 1;
    pws[0] = tfpow(ip[1]);
    pws[1] = tfpow(ip[2]);
    pws[2] = tfpow(ip[3]);

    float rx12 = 1. / (bus[1] - bus[1]);

    float yp[3];
    for (int idx = 0; idx < 3; ++idx) {
        float y1 = tftable(ib[0], ip[idx]);
        float y2 = tftable(ib[1], ip[idx]);
        yp[idx] = (y2 - y1) * rx12 * (bu - bus[0]) + y1;
    }

    float a1 = (qprime - pws[1]) * (qprime - pws[2]) / ((pws[0] - pws[1]) * (pws[0] - pws[2]));
    float a2 = (qprime - pws[0]) * (qprime - pws[2]) / ((pws[1] - pws[0]) * (pws[1] - pws[2]));
    float a3 = (qprime - pws[0]) * (qprime - pws[1]) / ((pws[2] - pws[0]) * (pws[2] - pws[1]));

    float deltf = a1 * yp[0] + a2 * yp[1] + a3 * yp[2];

    tf(l) = tm(l) + deltf + KELVIN;
}

void Feedback::updateTm(const int& l2d, const float* pow, int& nboiling) {

    float hlow = _hin;

    for (int k = 0; k < _g.nz(); ++k) {
        float hup = hlow + pow(l2d, k) / chflow(l2d);
        float havg = (hlow + hup) * 0.5;

        SteamError err = _steam.checkEnthalpy(havg);
        if (err == STEAM_TABLE_ERROR_MAXENTH) {
            nboiling = nboiling + 1;
            tm(l2d, k) = _steam.getSatTemperature();
        } else {
            tm(l2d, k) = _steam.getTemperature(havg);
            dm(l2d, k) = _steam.getDensity(havg);

        }
        hlow = hup;
    }
}

void
Feedback::setTfTable(const int& ntfpow, const int& ntfbu, const float* tfpow, const float* tfbu, const float* tftable) {
    _ntfpow = ntfpow;
    _ntfbu = ntfbu;
    _tfpow = new float[_ntfpow];
    _tfbu = new float[_ntfbu];
    _tftable = new float[_ntfbu * _ntfpow];

    copy(tfpow, tfpow + _ntfpow, _tfpow);
    copy(tfbu, tfbu + _ntfbu, _tfbu);
    copy(tftable, tfbu + _ntfbu * _ntfpow, _tftable);
}

