//
// Created by JOO IL YOON on 2021/01/30.
//

#include "BICGSolver.h"
#include "mat2g.h"
#include "myblas.h"

#define diag(igs, ige, l) diag[l*_g.ng2()+ige*_g.ng()+igs]
#define cc(lr, idir, ig, l) cc[l*_g.ng()*NDIRMAX*LR+ig*NDIRMAX*LR+idir*LR+lr]
#define src(ig, l) src[l*_g.ng()+ig]
#define aphi(ig, l) aphi[l*_g.ng()+ig]
#define b(ig, l) b[l*_g.ng()+ig]
#define x(ig, l) x[l*_g.ng()+ig]
#define phi(ig, l) phi[l*_g.ng()+ig]
#define b1d(ig, l)   b1d[(l*_g.ng())+ig]
#define x1d(ig, l)   x1d[(l*_g.ng())+ig]
#define b01d(ig, l)  _b01d[(l*_g.ng())+ig]
#define s1dl(ig, l)  _s1dl[(l*_g.ng())+ig]
#define b03d(ig, l)  _b03d[(l*_g.ng())+ig]
#define s3d(ig, l)  _s3d[(l*_g.ng())+ig]
#define s3dd(ig, l)  _s3dd[(l*_g.ng())+ig]


#define vr(ig, l)   _vr[(l*_g.ng())+ig]
#define vr0(ig, l)  _vr0[(l*_g.ng())+ig]
#define vp(ig, l)   _vp[(l*_g.ng())+ig]
#define vv(ig, l)   _vv[(l*_g.ng())+ig]
#define vs(ig, l)   _vs[(l*_g.ng())+ig]
#define vt(ig, l)   _vt[(l*_g.ng())+ig]
#define vy(ig, l)   _vy[(l*_g.ng())+ig]
#define vz(ig, l)   _vz[(l*_g.ng())+ig]
#define y1d(ig, l)   _y1d[(l*_g.ng())+ig]
#define b1i(ig, l)   _b1i[(l*_g.ng())+ig]

#define del(igs, ige, l)  _del[(l*_g.ng2())+(ige)*_g.ng()+(igs)]
#define ainvd(igs, ige, l)    _ainvd[(l*_g.ng2())+(ige)*_g.ng()+(igs)]
#define ainvl(igs, ige, l)    _ainvl[(l*_g.ng2())+(ige)*_g.ng()+(igs)]
#define ainvu(igs, ige, l)    _ainvu[(l*_g.ng2())+(ige)*_g.ng()+(igs)]
#define au(igs, ige, l)   _au[(l*_g.ng2())+(ige)*_g.ng()+(igs)]
#define delinv(igs, ige, l)   _delinv[(l*_g.ng2())+(ige)*_g.ng()+(igs)]
#define al(igs, ige, l)       _al[(l*_g.ng2())+(ige)*_g.ng()+(igs)]
#define deliau(igs, ige, l)   _deliau[(l*_g.ng2())+(ige)*_g.ng()+(igs)]


BICGSolver::BICGSolver(Geometry &g) : _g(g) {

    _vz = new SOL_VAR[_g.ng() * _g.nxyz()]{};
    _vy = new SOL_VAR[_g.ng() * _g.nxyz()]{};

    _vr = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _vr0 = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _vp = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _vv = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _vs = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _vt = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _y1d = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _b1i = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _b01d = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _s1dl = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _b03d = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _s3d = new CMFD_VAR[_g.ng() * _g.nxyz()]{};
    _s3dd = new CMFD_VAR[_g.ng() * _g.nxyz()]{};

    _del = new CMFD_VAR[_g.ng2() * _g.nxyz()]{};
    _ainvd = new CMFD_VAR[_g.ng2() * _g.nxyz()]{};
    _ainvl = new CMFD_VAR[_g.ng2() * _g.nxyz()]{};
    _ainvu = new CMFD_VAR[_g.ng2() * _g.nxyz()]{};
    _au = new CMFD_VAR[_g.ng2() * _g.nxyz()]{};
    _delinv = new CMFD_VAR[_g.ng2() * _g.nxyz()]{};
    _al = new CMFD_VAR[_g.ng2() * _g.nxyz()]{};
    _deliau = new CMFD_VAR[_g.ng2() * _g.nxyz()]{};
}

BICGSolver::~BICGSolver() {
    delete _vr;
    delete _vr0;
    delete _vp;
    delete _vv;
    delete _vs;
    delete _vt;
    delete _vy;
    delete _vz;
    delete _y1d;
    delete _b1i;
    delete _b01d;
    delete _s1dl;
    delete _b03d;
    delete _s3d;

    delete _del;
    delete _ainvd;
    delete _ainvl;
    delete _ainvu;
    delete _au;
    delete _delinv;
    delete _al;
    delete _deliau;
}

double BICGSolver::reset(const int &ig, const int &l, CMFD_VAR *diag, CMFD_VAR *cc, SOL_VAR *phi, CMFD_VAR *src) {
    double aphi = axb(ig, l, diag, cc, phi);
    vr(ig, l) = src(ig, l) - aphi;
    vr0(ig, l) = vr(ig, l);
    vp(ig, l) = 0.0;
    vv(ig, l) = 0.0;

    return vr(ig, l) * vr(ig, l);
}

void BICGSolver::reset(CMFD_VAR *diag, CMFD_VAR *cc, SOL_VAR *phi, CMFD_VAR *src, CMFD_VAR &r20) {

    _calpha = 1;
    _crho = 1;
    _comega = 1;

    r20 = 0;
    for (int l = 0; l < _g.nxyz(); ++l) {
        for (int ig = 0; ig < _g.ng(); ++ig) {
            r20 += reset(ig, l, diag, cc, phi, src);
        }
    }

    r20 = sqrt(r20);
}

void BICGSolver::sol1d(const int &j, const int &k, CMFD_VAR *b1d, CMFD_VAR *x1d) {

    int ibeg = _g.nxs(j);
    int iend = _g.nxe(j);
    int l = _g.ijtol(ibeg, j) + k * _g.nxy();

    //forward substitution
    matxvec2g(&delinv(0, 0, l), &b1d(0, l), &y1d(0, l));

    int lm1 = l;
    for (int i = ibeg + 1; i < iend; ++i) {
        l = l + 1;
        matxvec2g(&al(0, 0, l), &y1d(0, lm1), &b1i(0, l));
        for (int ig = 0; ig < _g.ng(); ++ig) {
            b1i(ig, l) = b1d(ig, l) - b1i(ig, l);
        }
        matxvec2g(&delinv(0, 0, l), &b1i(0, l), &y1d(0, l));
        lm1 = l;
    }

    //  backward substitution
    for (int ig = 0; ig < _g.ng(); ++ig) {
        x1d(ig, l) = y1d(ig, l);
    }

    int lp1 = l;
    for (int i = iend - 2; i >= ibeg; --i) {
        l = l - 1;
        matxvec2g(&deliau(0, 0, l), &x1d(0, lp1), &b1i(0, l));
        for (int ig = 0; ig < _g.ng(); ++ig) {
            x1d(ig, l) = y1d(ig, l) - b1i(ig, l);
        }
        lp1 = l;
    }
}

void BICGSolver::sol2d(CMFD_VAR *cc, const int &k, CMFD_VAR *b, CMFD_VAR *x) {
    //  forward solve

    for (int j = 0; j < _g.ny(); ++j) {
        for (int i = _g.nxs(j); i < _g.nxe(j); ++i) {
            int l = _g.ijtol(i, j) + k * _g.nxy();
            int ln = _g.neib(NORTH, l);
            if(ln > -1) {
                for (int ig = 0; ig < _g.ng(); ++ig) {
                    b01d(ig, l) = b(ig, l) - cc(LEFT, YDIR, ig, l) * s1dl(ig, ln);
                }
            } else {
                for (int ig = 0; ig < _g.ng(); ++ig) {
                    b01d(ig, l) = b(ig, l);
                }
            }
        }
        sol1d(j, k, _b01d, _s1dl);
        for (int i = _g.nxs(j); i < _g.nxe(j); ++i) {
            int l = _g.ijtol(i, j) + k * _g.nxy();
            for (int ig = 0; ig < _g.ng(); ++ig) {
                x(ig, l) = s1dl(ig, l);
            }
        }
    }

    //  backward solve
    for (int j = _g.ny() - 2; j >= 0; --j) {
        for (int i = _g.nxs(j); i < _g.nxe(j); ++i) {
            int l = _g.ijtol(i, j) + k * _g.nxy();
            int ln = _g.neib(SOUTH, l);
            for (int ig = 0; ig < _g.ng(); ++ig) {
                b01d(ig, l) = x(ig, ln) * cc(RIGHT, YDIR, ig, l);
            }
        }
        sol1d(j, k, _b01d, _s1dl);
        for (int i = _g.nxs(j); i < _g.nxe(j); ++i) {
            int l = _g.ijtol(i, j) + k * _g.nxy();
            for (int ig = 0; ig < _g.ng(); ++ig) {
                x(ig, l) -= s1dl(ig, l);
            }
        }
    }
}

void BICGSolver::minv(CMFD_VAR *cc, CMFD_VAR *b, SOL_VAR *x) {

    // forward solve
    for (int k = 0; k < _g.nz(); k++) {
        if (k == 0) {
            for (int l = k * _g.nxy(); l < (k + 1) * _g.nxy(); ++l) {
                for (int ig = 0; ig < _g.ng(); ++ig) {
                    b03d(ig, l) = b(ig, l);
                }
            }
        } else {
            for (int l = k * _g.nxy(); l < (k + 1) * _g.nxy(); ++l) {
                int lm = l - _g.nxy();
                for (int ig = 0; ig < _g.ng(); ++ig) {
                    b03d(ig, l) = b(ig, l) - cc(LEFT, ZDIR, ig, l) * s3d(ig, lm);
                }
            }
        }
        sol2d(cc, k, _b03d, _s3d);
        for (int l = k * _g.nxy(); l < (k + 1) * _g.nxy(); ++l) {
            for (int ig = 0; ig < _g.ng(); ++ig) {
                x(ig, l) = s3d(ig, l);
            }
        }
    }

    // backward solve
    for (int k = _g.nz() - 2; k >= 0; --k) {
        for (int l = k * _g.nxy(); l < (k + 1) * _g.nxy(); ++l) {
            int ln = _g.neib(TOP, l);
            for (int ig = 0; ig < _g.ng(); ++ig) {
                b03d(ig, l) = x(ig, ln) * cc(RIGHT, ZDIR, ig, l);
            }
        }
        sol2d(cc, k, _b03d, _s3d);
        for (int l = k * _g.nxy(); l < (k + 1) * _g.nxy(); ++l) {
            for (int ig = 0; ig < _g.ng(); ++ig) {
                x(ig, l) -= s3d(ig, l);
            }
        }
    }
}

void BICGSolver::abi1d(const int &j, const int &k) {
    // approximate block inverse from the LU factors
    int ix = _g.nxe(j) - 1;
    int l = _g.ijtol(ix, j) + k * _g.nxy();

    copyTomat2g(&delinv(0, 0, l), &ainvd(0, 0, l));

    for (int i = _g.nxe(j) - 2; i >= _g.nxs(j); --i) {
        int lp1 = l;
        l = l - 1;

        CMFD_VAR al1[4]{}, au1[4]{};;
        // lower part of the inverse
        matxmat2g(&ainvd(0, 0, lp1), &al(0, 0, lp1), al1); // lower part of the inverse
        al1[0] = -al1[0];
        al1[1] = -al1[1];
        al1[2] = -al1[2];
        al1[3] = -al1[3];
        matxmat2g(al1, &delinv(0, 0, l), &ainvl(0, 0, lp1)); // lower part of the inverse


        // upper part of the inverse
        matxmat2g(&delinv(0, 0, l), &au(0, 0, l), au1); // lower part of the inverse
        au1[0] = -au1[0];
        au1[1] = -au1[1];
        au1[2] = -au1[2];
        au1[3] = -au1[3];
        matxmat2g(au1, &ainvd(0, 0, lp1), &ainvu(0, 0, l)); // lower part of the inverse

        // diagonal part
        au1[0] = -au1[0];
        au1[1] = -au1[1];
        au1[2] = -au1[2];
        au1[3] = -au1[3];
        matxmat2g(au1, &ainvl(0, 0, lp1), &ainvd(0, 0, l)); // lower part of the inverse

        submat2g(&delinv(0,0,l), &ainvd(0, 0, l), &ainvd(0, 0, l));
    }
}

void BICGSolver::facilu1d(const int &j, const int &k) {
    int l = _g.ijtol(_g.nxs(j), j) + k * _g.nxy();

    // first column
    invmat2g(&del(0, 0, l), &delinv(0, 0, l));

    //   calc. inv(del)*u for later use in backsub
    matxmat2g(&delinv(0, 0, l), &au(0, 0, l), &deliau(0, 0, l));

    CMFD_VAR ald1[4]{}, temp[4]{};
    for (int i = _g.nxs(j) + 1; i < _g.nxe(j); ++i) {
        int lm1 = l;
        l = l + 1;
        matxmat2g(&al(0, 0, l), &delinv(0, 0, lm1), ald1);
        matxmat2g(ald1, &au(0, 0, lm1), temp);
        submat2g(&del(0,0,l), temp,&del(0,0,l));
        invmat2g(&del(0, 0, l), &delinv(0, 0, l));
        matxmat2g(&delinv(0, 0, l), &au(0, 0, l), &deliau(0, 0, l));
    }
}

void BICGSolver::facilu(CMFD_VAR *diag, CMFD_VAR *cc) {

//    &al = 0;
//    &au = 0;
    // loop over planes

    for (int k = 0; k < _g.nz(); ++k) {
        // first row
        int j = 0;
        for (int i = _g.nxs(j); i < _g.nxe(j); ++i) {
            int l = _g.ijtol(i, j) + k * _g.nxy();
            for (int ige = 0; ige < _g.ng(); ++ige) {
                for (int igs = 0; igs < _g.ng(); ++igs) {
                    del(igs, ige, l) = diag(igs, ige, l);
                }
                al(ige, ige, l) = cc(LEFT, XDIR, ige, l);
                au(ige, ige, l) = cc(RIGHT, XDIR, ige, l);
            }
        }

        //  === loop over rows ===
        for (int j = 1; j < _g.ny(); ++j) {
            int jm1 = j - 1;
            // obtain incomplete lu factor for the 1d matrix of the row
            facilu1d(jm1, k);
            // obtain the inverse of the 1d matrix
            abi1d(jm1, k);
            // d_j+1 = a_j+1 - l_j+1*jnv(d_j)*u_j
            for (int i = _g.nxs(j); i < _g.nxe(j); ++i) {
                int l = _g.ijtol(i, j) + k * _g.nxy();
                int ln = _g.neib(NORTH, l);
                if (ln > -1) {
                    CMFD_VAR ccy[LR][2]{}, temp[4]{};
                    ccy[LEFT][0] = cc(LEFT, YDIR, 0, l);
                    ccy[LEFT][1] = cc(LEFT, YDIR, 1, l);
                    ccy[RIGHT][0] = cc(RIGHT, YDIR, 0, ln);
                    ccy[RIGHT][1] = cc(RIGHT, YDIR, 1, ln);

                    diagxmat2g(ccy[LEFT], &ainvd(0, 0, ln), &del(0, 0, l));
                    matxdiag2g(&del(0, 0, l), ccy[RIGHT], temp);
                    submat2g(&diag(0, 0, l), temp, &del(0, 0, l));
                } else {
                    copyTomat2g(&diag(0, 0, l), &del(0, 0, l));
                }


                if (i == _g.nxs(j)) {
                    for (int ig = 0; ig < _g.ng(); ++ig) {
                        al(ig, ig, l) = cc(LEFT, XDIR, ig, l);
                    }
                } else {
                    int lw = _g.ijtol(i - 1, j) + k * _g.nxy();
                    int lwn = _g.neib(NORTH, lw);
                    if (lwn > -1) {
                        CMFD_VAR ccy[LR][2]{}, temp[4]{};
                        ccy[LEFT][0] = -cc(LEFT, YDIR, 0, l);
                        ccy[LEFT][1] = -cc(LEFT, YDIR, 1, l);
                        ccy[RIGHT][0] = cc(RIGHT, YDIR, 0, lwn);
                        ccy[RIGHT][1] = cc(RIGHT, YDIR, 1, lwn);

                        diagxmat2g(ccy[LEFT], &ainvl(0, 0, ln), temp);
                        matxdiag2g(temp, ccy[RIGHT], &al(0, 0, l));

                        for (int ig = 0; ig < _g.ng(); ++ig) {
                            al(ig, ig, l) += cc(LEFT, XDIR, ig, l);
                        }
                    } else {
                        for (int ig = 0; ig < _g.ng(); ++ig) {
                            al(ig, ig, l) = cc(LEFT, XDIR, ig, l);
                        }
                    }
                }

                if (i == _g.nxe(j) - 1) {
                    for (int ig = 0; ig < _g.ng(); ++ig) {
                        au(ig, ig, l) = cc(RIGHT, XDIR, ig, l);
                    }
                } else {
                    int le = _g.ijtol(i + 1, j) + k * _g.nxy();
                    int len = _g.neib(NORTH, le);
                    if (len > -1) {
                        CMFD_VAR ccy[LR][2]{}, temp[4]{};
                        ccy[LEFT][0] = -cc(LEFT, YDIR, 0, l);
                        ccy[LEFT][1] = -cc(LEFT, YDIR, 1, l);
                        ccy[RIGHT][0] = cc(RIGHT, YDIR, 0, len);
                        ccy[RIGHT][1] = cc(RIGHT, YDIR, 1, len);

                        diagxmat2g(ccy[LEFT], &ainvu(0, 0, ln), temp);
                        matxdiag2g(temp, ccy[RIGHT], &au(0, 0, l));

                        for (int ig = 0; ig < _g.ng(); ++ig) {
                            au(ig, ig, l) += cc(RIGHT, XDIR, ig, l);
                        }
                    } else {
                        for (int ig = 0; ig < _g.ng(); ++ig) {
                            au(ig, ig, l) = cc(RIGHT, XDIR, ig, l);
                        }
                    }
                }
            }
        }
        facilu1d(_g.ny() - 1, k);
    }
}

void BICGSolver::solve(CMFD_VAR *diag, CMFD_VAR *cc, CMFD_VAR &r20, SOL_VAR *phi, double &r2) {
    int n = _g.nxyz() * _g.ng();

    // solves the linear system by preconditioned BiCGSTAB Algorithm
    double crhod = _crho;
    _crho = myblas::dot(n, _vr0, _vr);
    _cbeta = _crho * _calpha / (crhod * _comega);

//    _vp(:,:,:)=_vr(:,:,:)+_cbeta*(_vp(:,:,:)-_comega*_vv(:,:,:))
    myblas::multi(n, _comega, _vv, _vt);
    myblas::minus(n, _vp, _vt, _vt);
    myblas::multi(n, _cbeta, _vt, _vt);
    myblas::plus(n, _vr, _vt, _vp);

    minv(cc, _vp, _vy);
    axb(diag, cc, _vy, _vv);

    CMFD_VAR r0v = myblas::dot(n, _vr0, _vv);

    if (r0v == 0.0) {
        return;
    }

    _calpha = _crho / r0v;

//    _vs(:,:,:)=_vr(:,:,:)-_calpha*_vv(:,:,:)
    myblas::multi(n, _calpha, _vv, _vt);
    myblas::minus(n, _vr, _vt, _vs);

    minv(cc, _vs, _vz);
    axb(diag, cc, _vz, _vt);

    CMFD_VAR pts = myblas::dot(n, _vs, _vt);
    CMFD_VAR ptt = myblas::dot(n, _vt, _vt);

    _comega = 0.0;
    if (ptt != 0.0) {
        _comega = pts / ptt ;
    }

//    phi(:, :, :) = phi(:, :, :) + _calpha * _vy(:,:,:)+_comega * _vz(:,:,:)
    myblas::multi(n, _comega, _vz, _vz);
    myblas::multi(n, _calpha, _vy, _vy);
    myblas::plus(n, _vz, _vy, _vy);
    myblas::plus(n, phi, _vy, phi);


//    _vr(:,:,:)=_vs(:,:,:)-_comega * _vt(:,:,:)
    myblas::multi(n, _comega, _vt, _vr);
    myblas::minus(n, _vs, _vr, _vr);

    if (r20 != 0.0) {
        r2 = sqrt(myblas::dot(n, _vt, _vt)) / r20;
    }
}

void BICGSolver::axb(CMFD_VAR *diag, CMFD_VAR *cc, SOL_VAR *phi, CMFD_VAR *aphi) {
    for (int l = 0; l < _g.nxyz(); ++l) {
        for (int ig = 0; ig < _g.ng(); ++ig) {
            aphi(ig,l) = axb(ig, l, diag, cc, phi);
        }
    }
}

double BICGSolver::axb(const int &ig, const int &l, CMFD_VAR *diag, CMFD_VAR *cc, SOL_VAR *phi) {
    double ab = 0.0;
    for (int igs = 0; igs < _g.ng(); ++igs) {
        ab += diag(igs, ig, l) * phi(igs, l);
    }

    for (int idir = 0; idir < NDIRMAX; ++idir) {
        for (int lr = 0; lr < LR; ++lr) {
            int ln = _g.neib(lr, idir, l);
            if (ln != -1)
                ab += cc(lr, idir, ig, l) * phi(ig, ln);
        }
    }

    return ab;
}
