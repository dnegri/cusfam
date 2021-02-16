#pragma once
#include "pch.h"


class SteamTable : public Managed {
protected:
	int _nprop = 6;
	int _np = 2;
	int  _npnts = 100;
	
	float* _press;

	float *_tmin
		, *_tmax
		, *_dmin
		, *_dmax
		, *_dgas
		, *_hmin
		, *_hmax
		, *_hgas
		, *_vismin
		, *_vismax
		, *_visgas
		, *_tcmin
		, *_tcmax
		, *_tcgas
		, *_shmin
		, *_shmax
		, *_shgas
		, *_rhdel
		, *_rtdel
		, *_rhdiff;

	float* _cmn, *_propc, *_hmod, *_dmodref;

public:
	__host__ SteamTable();
	__host__ virtual ~SteamTable();

	__host__ __device__ float getPressure();
	__host__ __device__ float getDensityByMartinelliNelson(const float& x);
	//float getDensity(const float& enthalpy);
	//float getTemperature(const float& enthalpy);
	//float getEnthalpy(const float& temperature);
	//float getSatTemperature();
	//int   checkEnthalpy(const float& enthalpy);
	__host__ virtual void setPressure(const float& press);

	__host__ __device__ void checkEnthalpy(const float& h, SteamError& err);
	__host__ __device__ void getSatTemperature(float& tm);
	__host__ __device__ void getTemperature(const float& h, float& tm);
	__host__ __device__ void getDensity(const float& h, float& dm);
	__host__ __device__ void getEnthalpy(const float& tm, float& h);
	__host__ __device__ void getRefDensity(const float& temperature, float& dm);


	__host__ __device__ float& propc(const int& type, const int& ip, const int& index) { return _propc[index*_np*_nprop + ip*_nprop + type]; }
	__host__ __device__ float& hmod(const int& ip, const int& index) { return _hmod[index * _np + ip ]; }
	__host__ __device__ float& dmodref(const int& ip, const int& index) { return _dmodref[index * _np + ip]; }
	__host__ __device__ float& cmn(const int& ip, const int& type) { return _cmn[type * 4 + ip]; }

	__host__ __device__ const float& press() const { return *_press; };
	__host__ __device__ const float& tmin() const { return *_tmin; };
	__host__ __device__ const float& tmax() const { return *_tmax; };
	__host__ __device__ const float& dmin() const { return *_dmin; };
	__host__ __device__ const float& dmax() const { return *_dmax; };
	__host__ __device__ const float& dgas() const { return *_dgas; };
	__host__ __device__ const float& hmin() const { return *_hmin; };
	__host__ __device__ const float& hmax() const { return *_hmax; };
	__host__ __device__ const float& hgas() const { return *_hgas; };
	__host__ __device__ const float& vismin() const { return *_vismin; };
	__host__ __device__ const float& vismax() const { return *_vismax; };
	__host__ __device__ const float& visgas() const { return *_visgas; };
	__host__ __device__ const float& tcmin() const { return *_tcmin; };
	__host__ __device__ const float& tcmax() const { return *_tcmax; };
	__host__ __device__ const float& tcgas() const { return *_tcgas; };
	__host__ __device__ const float& shmin() const { return *_shmin; };
	__host__ __device__ const float& shmax() const { return *_shmax; };
	__host__ __device__ const float& shgas() const { return *_shgas; };
	__host__ __device__ const float& rhdel() const { return *_rhdel; };
	__host__ __device__ const float& rtdel() const { return *_rtdel; };
	__host__ __device__ const float& rhdiff() const { return *_rhdiff; };
	__host__ __device__ const float* cmn() const { return _cmn; };
	__host__ __device__ const float* propc() const { return _propc; };
	__host__ __device__ const float* hmod() const { return _hmod; };
	__host__ __device__ const float* dmodref() const { return _dmodref; };
};




