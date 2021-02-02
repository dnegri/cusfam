#pragma once

#include "pch.h"

enum ChainType {
    IDX_DEFAULT,
    IDX_C,
    IDX_H,
    IDX_0,
};

enum ChainAction {
    LEAVE,
    UPDATE,
    HOLD
};

enum ChainReaction {
    R_CAP,
    R_DEC,
    R_N2N
};

class DepletionChain {
private:

    int mnucl;
    int nfcnt;

    int _nhvychn;
    int* _nheavy;                //(:)
    int _nhvyids;
    int* _hvyids;                //(:,:)
    int* _reactype;              //(:,:)
    int* _hvyupd;                //(:,:)

    // 2-1. Fission Product by Actinide Isotope
    int _nfiso;
    int _nfpiso;
    int* _fiso;
    int* _fpiso;           //(:)
    float* _fyld;


    // 2-2. Fission Product Chain Define
    int _nsm;
    int* _smids;         //(:)
    int _nxe;
    int* _xeids;          //(:)

    // 3. Decay Constant Define
    int _ndcy;
    int* _dcyID;         //(:)
    float* _dcy;              //(:)

public:
    DepletionChain();

    virtual ~DepletionChain();


};