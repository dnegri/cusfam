#pragma once

#include "pch.h"

class Nodal {
private:
	int _ng, _nxy, _nz, _ndir;
	float* d_trlcff0;
	float* d_jnet;

public:
	int nmaxswp;
	int nlupd;
	int nlswp;

public:
	virtual ~Nodal();

	void drive();
	void calculateTransverseLeakage();
};