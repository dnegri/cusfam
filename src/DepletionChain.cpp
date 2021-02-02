//
// Created by JOO IL YOON on 2021/02/01.
//

#include "DepletionChain.h"

DepletionChain::DepletionChain() {

    mnucl = 25;
    nfcnt = 12;

    _nhvychn = 4;
    _nheavy = new int[4]{10, 7, 8, 2};
    _nhvyids = 10 + 7 + 8 + 2;

    _hvyids = new int[_nhvyids]{
            U234, U235, U236, NP237, PU238, PU239, PU240, PU241, PU242, AM243,
            U238, NP239, PU239, PU240, PU241, PU242, AM243,
            U238, NP237, PU238, PU239, PU240, PU241, PU242, AM243,
            PU238, U234};

    _reactype = new int[_nhvyids]{
            R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP,
            R_CAP, R_CAP, R_DEC, R_CAP, R_CAP, R_CAP, R_CAP,
            R_CAP, R_N2N, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP, R_CAP,
            R_CAP, R_DEC};

    _hvyupd = new int[_nhvyids]{
            UPDATE, UPDATE, UPDATE, UPDATE, UPDATE, LEAVE, LEAVE, LEAVE, LEAVE, LEAVE,
            UPDATE, UPDATE, UPDATE, UPDATE, UPDATE, UPDATE, UPDATE,
            HOLD, LEAVE, LEAVE, LEAVE, LEAVE, LEAVE, LEAVE, LEAVE,
            HOLD, LEAVE};

    _nfiso = 12;
    _nfpiso = 5;
    _fiso = new int[_nfiso]{U234, U235, U236, U238, NP237,
                            NP239, PU238, PU239, PU240, PU241,
                            PU242, AM243};
    _fpiso = new int[_nfpiso]{PM147, PM149, SM149, I135, XE145};
    _fyld = new float[_nfiso * _nfpiso]{
            2.017740E-02, 1.035690E-02, 0.0, 4.901130E-02, 6.763670E-03,
            2.246730E-02, 1.081620E-02, 0.0, 6.281870E-02, 2.566345E-03,
            2.295290E-02, 1.338370E-02, 0.0, 5.974780E-02, 1.049093E-03,
            2.592740E-02, 1.625290E-02, 0.0, 6.940720E-02, 2.686420E-04,
            2.500000E-02, 1.547160E-02, 0.0, 6.903040E-02, 7.720750E-03,
            2.500000E-02, 1.547160E-02, 0.0, 6.903040E-02, 7.720750E-03,
            2.236530E-02, 1.596690E-02, 0.0, 5.740170E-02, 9.935130E-03,
            2.002960E-02, 1.216300E-02, 0.0, 6.541880E-02, 1.066411E-02,
            2.123450E-02, 1.393890E-02, 0.0, 6.731600E-02, 5.001020E-03,
            2.284950E-02, 1.474070E-02, 0.0, 6.943130E-02, 2.269029E-03,
            2.387710E-02, 1.598400E-02, 0.0, 7.388510E-02, 1.057970E-03,
            2.336130E-02, 1.555480E-02, 0.0, 6.034700E-02, 7.250690E-03
    };

    _nsm = 5;
    _smids = new int[_nsm]{PM147, PS148, PM148, PM149, SM149};
    _nxe = 2;
    _xeids = new int[_nxe]{I135, XE145};


    _ndcy = 9;
    _dcyID = new int[_ndcy]{NP239, PU241, PU238, PM147, PS148, PM148, PM149, I135, XE145};
    _dcy = new float[_ndcy]{3.40515E-06, 1.53705E-09, 2.50451E-10,
                           8.37254E-09, 1.49451E-06, 1.94297E-07, 3.62737E-06,
                           2.93061E-05, 2.10657E-05};
}

DepletionChain::~DepletionChain() {
    delete [] _nheavy;
    delete [] _hvyids;
    delete [] _reactype;
    delete [] _hvyupd;
    delete [] _fiso;
    delete [] _fpiso;
    delete [] _fyld;
    delete [] _smids;
    delete [] _xeids;
    delete [] _dcyID;
    delete [] _dcy;
}


