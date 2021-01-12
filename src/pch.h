#ifndef PCH_H_
#define PCH_H_

#include <array>
#include <map>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifndef CPU
    #include "helper_cuda.h"
    #include <cuda_runtime.h>
#else
    #define __global__
    #define __device__
#endif


static const int		PLUS = 1;
static const int		MINUS = -1;

static const int		XDIR = 0;
static const int		YDIR = 1;
static const int		ZDIR = 2;
static const int	    NDIRMAX = 3;


static const int		LEFT = 0;
static const int		RIGHT = 1;
static const int		LEFTRIGHT = 2;
static const int		LR = 2;
static const int		CENTER = 2;
static const int		LRC = 3;

static const int		UP = 0;
static const int		DOWN = 1;
static const int		NUPDOWN = 2;


static const int		FORWARD = 0;
static const int		BACKWARD = 1;
static const int		NFORBACK = 2;

static const int	 	POSITIVE = 0;
static const int		NEGATIVE = 1;
static const int	 	NSLOPE = 2;

static const int		SELF = 0;
static const int		NEIB = 1;
static const int		SELFNEIB = 2;


static const int		NW = 0;
static const int		NE = 1;
static const int		SE = 2;
static const int		SW = 3;

static const int		WEST = 0;
static const int		EAST = 1;
static const int		NORTH = 2;
static const int		SOUTH = 3;
static const int		BOT= 4;
static const int		TOP = 5;
static const int		NEWS = 4;
static const int		NEWSBT = 6;
static const int	    NEWS2XY[NEWS] = { XDIR, XDIR, YDIR, YDIR};

static const int		NDIVREG = 8;
static const double RDIVREG = 1.0 / NDIVREG;

static const double PI = 3.141592;
static const double BIG = 1.E+30;

static const double MICRO = 1.E-6;
static const double MILLI = 1.E-3;

static const int    NG2 = 2;

using namespace std;

static const int    NTHREADSPERBLOCK = 64;

#define var3(var,ig,l,k)        var[(k*_nxy+l)*_ng+ig]
#define var4(var,igs,igd,l,k)   var[((k*_nxy+l)*_ng+igs)*_ng+igd]
#endif /* PCH_H_ */
