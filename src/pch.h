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
#else
    #define __global__
    #define __device__
    #define __host__
    #define __constant__
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

__constant__  static const int    NG2 = 2;

using namespace std;

__constant__  static const int    NTHREADSPERBLOCK = 64;

#define var3(var,ig,l,k)        var[(k*_nxy+l)*_ng+ig]
#define var4(var,igs,igd,l,k)   var[((k*_nxy+l)*_ng+igs)*_ng+igd]
#endif /* PCH_H_ */
