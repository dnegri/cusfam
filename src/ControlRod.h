#pragma once


#include "Geometry.h"

enum CEA {
    P,
    R1,
    R2,
    R3,
    R4,
    R5,
    SA,
    SB,

};

class ControlRod {
    Geometry* _g;
    float* _ratio;
    CEA* _cea;

public:
    ControlRod(Geometry& g);

    virtual ~ControlRod();

    void setPosition(const CEA& rodid, const int& pos);

    void setPosition(const int& rodidx, const int& pos);

    const float& ratio(const int& l) {return _ratio[l];};
    const CEA& cea(const int& l) {return _cea[l];};

};


