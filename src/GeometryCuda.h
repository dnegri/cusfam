#pragma once
#include "pch.h"
#include "Geometry.h"

/**
 * Nodal Geometry
 * The naming rule for indexing variables
 *
 * the number of nodes : n+(x,y)+(a)+(f)
 * n    : number
 * x    : x-direction
 * y    : y-direction
 * xy   : 2D(xy) plane
 * a    : asembly-wise
 * f    :: fuel only
 *
 * one-dimensional indices : (i,j,k)+(s,e)+(a)+(f)
 * i    : x-direction
 * j    : y-direction,
 * k    : z-direction
 * s    : starting
 * e    : ending
 * a    : assembly-wise
 * f    : fuel only
 *
 * note that indices follows the c-style index numbering
 * starting index   : 0     (included)
 * ending index     : n+1   (not included)
 *
 * Two-dimensional indices
 * la   : assembly index in 2D
 * l    : node index in 2D
 *
 */





class GeometryCuda : public Geometry {

public:
	__host__ GeometryCuda(const Geometry& g);
	__host__ virtual ~GeometryCuda();
};

