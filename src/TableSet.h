#pragma once
#include "pch.h"

class TableSet : public Managed {

public:
	void readTableSet(const int& ncomp, const int* icomps, const float* xs, const float* xsd);
	void readTableSet(const int* icomp, const float* xs, const float* xsd);


};
