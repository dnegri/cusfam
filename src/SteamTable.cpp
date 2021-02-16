#include "SteamTable.h"

extern "C" {
	void copySteamTable(
          float* press
		, float* tmin
		, float* tmax
		, float* dmin
		, float* dmax
		, float* dgas
		, float* hmin
		, float* hmax
		, float* hgas
		, float* vismin
		, float* vismax
		, float* visgas
		, float* tcmin
		, float* tcmax
		, float* tcgas
		, float* shmin                    
		, float* shmax
		, float* shgas
		, float* rhdel
		, float* rtdel
		, float* rhdiff
		, float* cmn
		, float* propc
		, float* hmod
		, float* dmodref);
}

SteamTable::SteamTable()
{
    _press = new float;
    _tmin = new float;
    _tmax = new float;
    _dmin = new float;
    _dmax = new float;
    _dgas = new float;
    _hmin = new float;
    _hmax = new float;
    _hgas = new float;
    _vismin = new float;
    _vismax = new float;
    _visgas = new float;
    _tcmin = new float;
    _tcmax = new float;
    _tcgas = new float;
    _shmin = new float;
    _shmax = new float;
    _shgas = new float;
    _rhdel = new float;
    _rtdel = new float;
    _rhdiff = new float;


	_cmn = new float[2 * 4]{};
	_hmod = new float[_np*_npnts]{};
	_dmodref = new float[_np * _npnts]{};
	_propc = new float[_nprop*_np*_npnts] {};
}

SteamTable::~SteamTable()
{
}

void SteamTable::setPressure(const float& press) {
	*_press = press;

	copySteamTable(_press
		, _tmin
		, _tmax
		, _dmin
		, _dmax
		, _dgas
		, _hmin
		, _hmax
		, _hgas
		, _vismin
		, _vismax
		, _visgas
		, _tcmin
		, _tcmax
		, _tcgas
		, _shmin
		, _shmax
		, _shgas
		, _rhdel
		, _rtdel
		, _rhdiff
		, _cmn
		, _propc
		, _hmod
		, _dmodref);
}


float SteamTable::getPressure() {
	return *_press;
}

void SteamTable::getDensity(const float& h, float& dm) {
    int index;

    if (h <= hmax()) {
        index = (h - hmin()) * rhdel();
        index = max(index, 0);
        index = min(index, _npnts - 1);

        dm = propc(PROP_DENS, 0, index) * h + propc(PROP_DENS, 1, index);
        dm = max(dm, dmax());
    }
    else {
        float x = (h - hmax()) * rhdiff();

        //modified martinelli - nelson
        dm = getDensityByMartinelliNelson(x);
    }
}

void SteamTable::getRefDensity(const float& tm, float& dm) {

    int index;

    index = (tm - tmin()) * rtdel();
    index = max(index, 0);
    index = min(index, _npnts - 1);

    dm = dmodref(0, index) * tm + dmodref(1, index);

    dm = max(dm, dmax());
}


float SteamTable::getDensityByMartinelliNelson(const float& x) {

    static const float PRESS_BOUNDARY = 127.55;
    float alpha; // void fraction

    if (*_press < PRESS_BOUNDARY) {
        if (x < 0.01) {
            alpha = 0.0;
        }
        else if (x < 0.1) {
            alpha = cmn(0, 0) + cmn(1, 0) * x + cmn(2, 0) * x * x + cmn(3, 0) * x * x * x;
        }
        else if (x < 0.9) {
            alpha = cmn(0, 1) + cmn(1, 1) * x + cmn(2, 1) * x * x + cmn(3, 1) * x * x * x;
        }
        else {
            alpha = 1.0;
        }
    }
    else {
        alpha = x / dgas() / ((1.0 - x) / dmax() + x / dgas());
    }

    float dm = (1. - alpha) * dmax() + alpha * dgas();
    return dm;
}

void SteamTable::getTemperature(const float& h, float& tm) {

    int index;

    index = (h - hmin()) * rhdel();
    index = max(index, 0);
    index = min(index, _npnts - 1);

    tm = propc(PROP_TEMP, 0, index) * h + propc(PROP_TEMP, 1, index);
    tm = min(tm, tmax());
}


void SteamTable::getEnthalpy(const float& tm, float& h) {
    int index;

    index = (tm - tmin()) * rtdel();
    index = max(index, 0);
    index = min(index, _npnts - 1);

    h = hmod(0, index) * tm + hmod(1, index);

    h = min(h, hmax());
}

void SteamTable::getSatTemperature(float& tm) {

    tm = tmax();

}

void SteamTable::checkEnthalpy(const float& h, SteamError& err) {
    SteamError ierr;

    if (h > hmax()) {
        ierr = STEAM_TABLE_ERROR_MAXENTH;
    }
    else {
        ierr = NO_ERROR;
    }

    err = ierr;
}
