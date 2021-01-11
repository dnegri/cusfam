#pragma once
#include "pch.h"
#include "Geometry.h"


class Nodal {
protected:
	Geometry& _g;

public:
	int nmaxswp;
	int nlupd;
	int nlswp;

public:
	Nodal(Geometry& g);
	virtual ~Nodal();


};