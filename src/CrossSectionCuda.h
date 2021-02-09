#pragma once
#include "pch.h"
#include "CrossSection.h"

class CrossSectionCuda : public CrossSection
{
public:
	CrossSectionCuda(const CrossSection& x);
	void updateXS(const float* dnst, const float* dppm, const float* dtf, const float* dtm);
};

