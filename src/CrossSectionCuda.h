#pragma once
#include "pch.h"
#include "CrossSection.h"

class CrossSectionCuda : public CrossSection
{
public:
	__host__ CrossSectionCuda(const CrossSection& x);
	__host__ void copyXS(const CrossSection& x);
	__host__ void updateXS(const float* dnst, const float* dppm, const float* dtf, const float* dtm);
	__host__ void updateMacroXS(float* dnst);
};

