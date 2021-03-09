#pragma once
#include "Geometry.h"

static const int LEN_ROD_NAME = 5;
static const int MAX_ROD_TYPE = 73;
static const float EPS_ROD_IN = 0.01;


class ControlRod {
    Geometry* _g;
    float* _ratio;
    int* _ceamap;
	int _ncea;
	char _idcea[MAX_ROD_TYPE][LEN_ROD_NAME];
	int _abstype[MAX_ROD_TYPE];

public:
    ControlRod(Geometry& g);
    virtual ~ControlRod();
	
	void initialize(const int& ncea, const int* iabs, const char* idcea[], const int* ceamap);

    void setPosition(const char* rodid, const int& pos);

    void setPosition(const int& rodidx, const int& pos);

    const float& ratio(const int& l) {return _ratio[l];};
    const int& cea(const int& l) {return _ceamap[l];};
	int* ceamap() { return _ceamap; };

	int& ncea() { return _ncea; };
	char* idcea(const int& icea) { return _idcea[icea]; };
	char** idcea() { return (char**)_idcea; };
	int* abstype() { return _abstype; };
	int& abstype(const int& icea) { return _abstype[icea]; }
};


