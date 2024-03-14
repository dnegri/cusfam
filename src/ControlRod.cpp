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

void ControlRod::setDefaultPosition()
{
	// top reflector insertion by default
	for (int l2d = 0; l2d < _g->nxy(); l2d++)
	{
		int icea = cea(l2d);

		if (icea == -1) continue;

		int l = _g->kec() * _g->nxy() + l2d;
		_ratio[l] = 1.0;
	}
}

void ControlRod::setPosition(const char* rodid, const float& pos) {


	for (int l2d = 0; l2d < _g->nxy(); l2d++)
	{	
		int icea = cea(l2d);
		bool found = strncmp(rodid, idcea(icea), strlen(rodid)) == 0;

		if (!found) continue;

		int l = l2d;

		float zpos = 0.0;
		//for (int k = 0; k < _g->kbc(); k++) {
		//	zpos += _g->hmesh(ZDIR, l);
		//}

		int k = _g->kbc();
		l += k * _g->nxy();

		for (; k < _g->kec(); k++)
		{
			float hz = _g->hz(k);
			zpos += hz;

			if (zpos - EPS_ROD_IN >= pos) {
				_ratio[l] = (zpos - pos) / hz;
				l += _g->nxy();
				break;
			}
			else {
				_ratio[l] = 0.0;
				l += _g->nxy();
			}
		}

		for (++k; k < _g->kec(); k++) {
			_ratio[l] = 1.0;
			l += _g->nxy();
		}

		float rodLengthIn = _g->hzcore() - pos;
		float lengthOfAbsorption = _g->hzcore(); // assumption absorption length = _g->hzcore()
		float rodLengthOut = lengthOfAbsorption - rodLengthIn;

		float ratioTopRefl = min(1.0, rodLengthOut / _g->hz(_g->kec())); // rod in ration in top reflector 
		_ratio[l] = ratioTopRefl;
	}
}

void ControlRod::setPosition(const int& rodidx, const float& pos) {

}
