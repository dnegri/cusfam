#pragma once

class CMFD {
	int _ng;


	float* dtilz, dhatz, dtilr, dhatr;
	float* am, af;
	float* ccz, ccr;
	float* src;

public:
	void upddtil(const int& ls);

};