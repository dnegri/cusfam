#ifndef PCH_H_
#define PCH_H_

#include <array>
#include <map>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>

using namespace std;

#define CPU

#ifndef CPU
    #include <cuda_runtime.h>
    #include "helper_string.h"
    #include "helper_cuda.h"
    class Managed {
    public:
        void* operator new(int len) {
            void* ptr;
            cudaMallocManaged(&ptr, len);
            checkCudaErrors(cudaDeviceSynchronize());
            return ptr;
        }

        void operator delete(void* ptr) {
            checkCudaErrors(cudaDeviceSynchronize());
            cudaFree(ptr);
        }
    };
    
    __constant__ const int  NTHREADSPERBLOCK=1024;

    extern dim3 BLOCKS_NGXYZ;
    extern dim3 THREADS_NGXYZ;
    extern dim3 BLOCKS_NODE;
    extern dim3 THREADS_NODE;
    extern dim3 BLOCKS_2D;
    extern dim3 THREADS_2D;
    extern dim3 BLOCKS_SURFACE;
    extern dim3 THREADS_SURFACE;

#else
    #define __global__
    #define __device__
    #define __host__
    #define __constant__

    class Managed {

    };
#endif


__constant__  static const int		PLUS = 1;
__constant__  static const int		MINUS = -1;

__constant__  static const int		XDIR = 0;
__constant__  static const int		YDIR = 1;
__constant__  static const int		ZDIR = 2;
__constant__  static const int	    NDIRMAX = 3;


__constant__  static const int		LEFT = 0;
__constant__  static const int		RIGHT = 1;
__constant__  static const int		LEFTRIGHT = 2;
__constant__  static const int		LR = 2;
__constant__  static const int		CENTER = 2;
__constant__  static const int		LRC = 3;

__constant__  static const int		UP = 0;
__constant__  static const int		DOWN = 1;
__constant__  static const int		NUPDOWN = 2;


__constant__  static const int		FORWARD = 0;
__constant__  static const int		BACKWARD = 1;
__constant__  static const int		NFORBACK = 2;

__constant__  static const int	 	POSITIVE = 0;
__constant__  static const int		NEGATIVE = 1;
__constant__  static const int	 	NSLOPE = 2;

__constant__  static const int		SELF = 0;
__constant__  static const int		NEIB = 1;
__constant__  static const int		SELFNEIB = 2;


__constant__  static const int		NW = 0;
__constant__  static const int		NE = 1;
__constant__  static const int		SE = 2;
__constant__  static const int		SW = 3;

__constant__  static const int		WEST = 0;
__constant__  static const int		EAST = 1;
__constant__  static const int		NORTH = 2;
__constant__  static const int		SOUTH = 3;
__constant__  static const int		BOT= 4;
__constant__  static const int		TOP = 5;
__constant__  static const int		NEWS = 4;
__constant__  static const int		NEWSBT = 6;
__constant__  static const int	    NEWS2XY[NEWS] = { XDIR, XDIR, YDIR, YDIR};

__constant__  static const int		NDIVREG = 8;
__constant__  static const float RDIVREG = 1.0 / NDIVREG;

__constant__  static const float PI = 3.141592;
__constant__  static const float BIG = 1.E+30;

__constant__  static const float MICRO = 1.E-6;
__constant__  static const float MILLI = 1.E-3;
__constant__  static const float KELVIN = 273.15;

__constant__  static const int    NG2 = 2;

__constant__ static const int NISO = 40;
__constant__ static const int NDEP = 25;

__constant__ static const char* ISOTOPE_NAME[NISO] = {
        "U234", "U235", "U236", "NP37", "U238",
        "PU48", "NP39", "PU49", "PU40", "PU41",
        "PU42", "AM43", "RESI", "POIS", "PM47",
        "PS48", "PM48", "PM49", "SM49", "I135",
        "XE45", "FP.1", "SB10", "H2O ", "STRM",
        "AM41", "AM42", "CM42", "CM44", "TH32",
        "PA33", "U233", "MAC ", "DEL1", "DEL2",
        "DEL3", "TMOD", "DETE", "   V", "XSE " };


enum Isotope {
    U234, U235, U236, NP37, U238,
    PU48, NP39, PU49, PU40, PU41,
    PU42, AM43, RESI, POIS, PM47,
    PS48, PM48, PM49, SM49, I135,
    XE45, FP1, SB10, H2O, STRM,
    AM41, AM42, CM42, CM44, TH32,
    PA33, U233, MAC, DEL1, DEL2,
    DEL3, TMOD, DETE, V, XSE
};

__constant__ static const int NHEAVY = 12;
__constant__ static const int ISOHVY[]{ U234, U235, U236, NP37, U238,
                                        PU48, NP39, PU49, PU40, PU41,
                                        PU42, AM43 };
__constant__ static const int NMAC = 16;
__constant__ static const int ISOMAC[]{ U234, U235, U236, NP37, U238, 
                                        PU48, NP39, PU49, PU40, PU41, 
                                        PU42, AM43, RESI, POIS, FP1 , STRM};

//  9 - the number of isotopes(pm47, ps48, pm48, pm49, sm, i135, xenon, b10, h2o)
__constant__ static const int NNIS = 9;
__constant__ static const int ISONIS[]{ PM47, PS48, PM48, PM49, SM49, 
                                        I135, XE45, SB10, H2O};

__constant__ static const int NDCY = 9;
__constant__ static const int ISODCY[]{NP39, PU41, PU48, PM47, PS48,
                                       PM48, PM49, I135, XE45};
__constant__ static const float DCY[] {3.40515E-06, 1.53705E-09, 2.50451E-10, 8.37254E-09, 1.49451E-06,
                                       1.94297E-07, 3.62737E-06, 2.93061E-05, 2.10657E-05};


__constant__ static const int NFP = 4;
enum ISO_FP {IFP_PM47, IFP_PM49, IFP_I135, IFP_XE45};
__constant__ static const int ISOFP[]{ PM47, PM49, I135, XE45 };
__constant__ static const float FPYLD[]{
            2.017740E-02, 1.035690E-02, 4.901130E-02, 6.763670E-03,
            2.246730E-02, 1.081620E-02, 6.281870E-02, 2.566345E-03,
            2.295290E-02, 1.338370E-02, 5.974780E-02, 1.049093E-03,
            2.500000E-02, 1.547160E-02, 6.903040E-02, 7.720750E-03,
            2.592740E-02, 1.625290E-02, 6.940720E-02, 2.686420E-04,
            2.236530E-02, 1.596690E-02, 5.740170E-02, 9.935130E-03,
            2.500000E-02, 1.547160E-02, 6.903040E-02, 7.720750E-03,
            2.002960E-02, 1.216300E-02, 6.541880E-02, 1.066411E-02,
            2.123450E-02, 1.393890E-02, 6.731600E-02, 5.001020E-03,
            2.284950E-02, 1.474070E-02, 6.943130E-02, 2.269029E-03,
            2.387710E-02, 1.598400E-02, 7.388510E-02, 1.057970E-03,
            2.336130E-02, 1.555480E-02, 6.034700E-02, 7.250690E-03};

__constant__ static const int NPTM = 2;

__constant__ static const float HAW = 1.0079;
__constant__ static const float OAW = 15.994915;
__constant__ static const float H2OAW = 18.010715;

__constant__ static const float AVOG = 0.6022045;
__constant__ static const float B10AW = 10.012937;
__constant__ static const float B11AW = 11.009305;

__constant__ static const int TF_POINT = 20;

enum SteamError {
    NO_ERROR,
    STEAM_TABLE_ERROR_MAXENTH
};

enum PROP_TYPE {
    PROP_TEMP,
    PROP_ENTH,
    PROP_DENS,
    PROP_VISC,
    PROP_TCON,
    PROP_SPCH
};

#define var3(var,ig,l,k)        var[(k*_nxy+l)*_ng+ig]
#define var4(var,igs,igd,l,k)   var[((k*_nxy+l)*_ng+igs)*_ng+igd]

#define NODAL_VAR float

#define CMFD_VAR    float
#define SOL_VAR    float

#endif /* PCH_H_ */
