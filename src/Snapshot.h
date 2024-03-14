#pragma once

#include "Depletion.h"
#include "Geometry.h"
#include "pch.h"

class Snapshot {
private:
    Geometry &_g;
    float *_dnst;
    float *_burn;

public:
    Snapshot(Geometry &g) : _g(g) {
        _dnst = new float[NISO * g.nxyz()]{};
        _burn = new float[g.nxyz()]{};
    }
    ~Snapshot(){};

    void save(Simon &s) {
        copy(s.d().dnst(), s.d().dnst() + NISO * _g.nxyz(), _dnst);
        copy(s.d().burn(), s.d().burn() + _g.nxyz(), _burn);
    }

    void load(Simon &s) {
        copy(_dnst, _dnst + NISO * _g.nxyz(), s.d().dnst());
        copy(_burn, _burn + _g.nxyz(), s.d().burn());
    }
};
