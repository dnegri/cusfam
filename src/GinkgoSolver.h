#pragma once


#include "BICGCMFD.h"

class GinkgoSolver : public Managed {
private:
    Geometry& _g;
    int * _idx_col;
    int * _idx_row;
    float * _a;
public:
    GinkgoSolver(Geometry& g);

    virtual ~GinkgoSolver();

    void initialize();

};


