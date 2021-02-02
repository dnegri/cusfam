#ifndef PCH_H_
#define PCH_H_

#include <array>
#include <map>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifndef CPU
    #include <cuda_runtime.h>
    #include "helper_string.h"
    #include "helper_cuda.h"
    class Managed {
    public:
        void* operator new(size_t len) {
            void* ptr;
            cudaMallocManaged(&ptr, len);
            cudaDeviceSynchronize();
            return ptr;
        }

        void operator delete(void* ptr) {
            cudaDeviceSynchronize();
            cudaFree(ptr);
        }
    };
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
__constant__  static const double RDIVREG = 1.0 / NDIVREG;

__constant__  static const double PI = 3.141592;
__constant__  static const double BIG = 1.E+30;

__constant__  static const double MICRO = 1.E-6;
__constant__  static const double MILLI = 1.E-3;
__constant__  static const float KELVIN = 273.15;

__constant__  static const int    NG2 = 2;

using namespace std;

__constant__  static const int    NTHREADSPERBLOCK = 64;

__constant__ static const int MAX_ISOTOPE = 31;
__constant__ static const int NUM_FISSION = 16;
__constant__ static const int NUM_YIELD = 7;
__constant__ static const int NUM_DECAY = 9;
__constant__ static const int NUM_POISON = 1;
__constant__ static const int LEN_ISONAME = 5;

__constant__ const char* _ISOTOPE_NAME[MAX_ISOTOPE] = {
        "U234 ", "U235 ", "U236 ", "U238 ", "NP237",
        "NP239", "PU238", "PU239", "PU240", "PU241",
        "PU242", "AM241", "AM242", "AM243", "CM242",
        "CM244", "POIS ", "SB10 ", "H2O  ", "MAC  ",
        "PM147", "PS148", "PM148", "PM149", "SM149",
        "I135 ", "XE145", "XSE  ", "DEL1 ", "DEL2 ",
        "DEL3"};


enum Isotope {
    U234 , U235 , U236 , U238 , NP237,
    NP239, PU238, PU239, PU240, PU241,
    PU242, AM241, AM242, AM243, CM242,
    CM244, POIS , SB10 , H2O  , MAC  ,
    PM147, PS148, PM148, PM149, SM149,
    I135 , XE145, XSE  , DEL1 , DEL2 ,
    DEL3
};


//  9 - the number of isotopes(pm47, ps48, pm48, pm49, sm, i135, xe, b10, h2o)
__constant__ static const int NNIS = 9;

static const int ISONIS[]{PM147, PS148, PM148, PM149, SM149, I135, XE145, SB10, H2O};

#define var3(var,ig,l,k)        var[(k*_nxy+l)*_ng+ig]
#define var4(var,igs,igd,l,k)   var[((k*_nxy+l)*_ng+igs)*_ng+igd]



#endif /* PCH_H_ */
