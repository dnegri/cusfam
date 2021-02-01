#pragma once

#include "pch.h"

enum Isotope {

};
enum ChainType {
    IDX_DEFAULT,
    IDX_C,
    IDX_H,
    IDX_0,
};

enum ChainAction {
    DO_NOTHING,
    DO_UPDATE,
    DO_HOLD
};

enum ChainReaction {
    R_CAP,
    R_DEC,
    R_N2N
};

struct Chain {
    int niso;
    ChainReaction* reactions;
    ChainAction* actions;
    Isotope* isotope;
};

class DepletionChain {
private:
    static const int _MAX_ISOTOPE = 31;
    static const int _NUM_FISSION = 16;
    static const int _NUM_YIELD = 7;
    static const int _NUM_DECAY = 9;
    static const int _NUM_POISON = 1;
    static const int _LEN_ISONAME = 5;


    const char* _ISOTOPE_NAME[_MAX_ISOTOPE] = {
            "U234 ", "U235 ", "U236 ", "U238 ", "NP237",
            "NP239", "PU238", "PU239", "PU240", "PU241",
            "PU242", "AM241", "AM242", "AM243", "CM242",
            "CM244", "POIS ", "SB10 ", "H2O  ", "MAC  ",
            "PM147", "PS148", "PM148", "PM149", "SM149",
            "I135 ", "XE145", "XSE  ", "DEL1 ", "DEL2 ",
            "DEL3"};



    int* _nHeavyChain;                //(:)
    Chain* heavyChain;

    // 2-1. Fission Product by Actinide Isotope
    float* _fyld;             //(:,:;)(NUM_YIELD,NUM_FISSION)
    int _ActinideNum
    int _fpIsotopeNum
    int* _ActinideID;            //(:)
    int* _fpIsotopeID;           //(:)

    // 2-2. Fission Product Chain Define
    int _nsm;
    int* _smids;         //(:)
    int _xenonIsoNum;
    int* _xenonChainID;          //(:)

    // 3. Decay Constant Define
    int _decayIsotopeNum;
    int* _dcyID;         //(:)
    float* _dcy;              //(:)

public:
    DepletionChain();

    virtual ~DepletionChain();


};