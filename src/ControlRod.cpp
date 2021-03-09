//
// Created by JOO IL YOON on 2021/02/13.
//

#include "ControlRod.h"

ControlRod::ControlRod(Geometry& g) {
    _g = &g;
	_ceamap = new int[_g->nxy()];
	_ratio = new float[_g->nxyz()]{};
}

ControlRod::~ControlRod() {

}

void ControlRod::setPosition(const char* rodid, const int& pos) {


	for (int l2d = 0; l2d < _g->nxy(); l2d++)
	{	
		int icea = cea(l2d);
		bool found = strncmp(rodid, idcea(icea), strlen(rodid)) == 0;

		if (!found) continue;

		int l = l2d;
		float zpos = 0.0;

		int k = 0;
		for (; k < _g->nz(); k++)
		{
			float hz = _g->hmesh(ZDIR, l);
			zpos += hz;

			if (zpos + EPS_ROD_IN >= pos) {
				_ratio[l] = (zpos - pos) / hz;
				break;
			}
			else {
				_ratio[l] = 0.0;
			}


			l += _g->nxy();
		}

		for (; k < _g->nz(); k++) {
			l += _g->nxy();
			_ratio[l] = 1.0;
		}

	}
}

void ControlRod::setPosition(const int& rodidx, const int& pos) {

}
