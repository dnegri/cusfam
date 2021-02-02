#pragma once
#include "pch.h"

#ifndef XS_PRECISION
    #define XS_PRECISION double
#endif

class CrossSection : public Managed {
private:
    int _ng;
    int _nxyz;
    int _mnucl;
    int _nptm;

    XS_PRECISION* _xsnf;
    XS_PRECISION* _xsdf;
    XS_PRECISION* _xssf;
    XS_PRECISION* _xstf;
    XS_PRECISION* _xskf;
    XS_PRECISION* _chif;
    XS_PRECISION* _xsadf;

//
// pointers for micro cross section derivative for isotopes
//
// xd*micd    transport
// xd*mica    absorption
// xd*mics    scatter
// xd*micf    fission
// xd*mick    kappa-fission
// xd*micn    nue-fission
//
// * = p for ppm derivatives
//   = m for moderator temperature derivatives
//   = d for moderator density derivatives
//   = f for fuel temperature derivatives
//
//
// (1) = ipm49th
// (2) = isamth
// (3) = ii135th
// (4) = ixenth
// (5) = ib10th
// (6) = ih2oth
//
// xdpmicf = pointers for micro xs ppm derivatives for fission nuclides (nt2d,nz,ng,nfcnt)
// xdfmicf = pointers for micro xs tf  derivatives for fission nuclides (nt2d,nz,ng,nfcnt)
// xdmmicf = pointers for micro xs tm  derivatives for fission nuclides (nt2d,nz,ng,nfcnt)
// xddmicf = pointers for micro xs dm  derivatives for fission nuclides (nt2d,nz,ng,nfcnt)
//
//
    XS_PRECISION* _xdpmicf;   // (:,:,:,:,:)
    XS_PRECISION* _xdfmicf;   // (:,:,:,:,:)
    XS_PRECISION* _xdmmicf;   // (:,:,:,:,:)
    XS_PRECISION* _xddmicf;   // (:,:,:,:,:)
    XS_PRECISION* _xdfmics;   // (:,:,:,:,:,:)
    XS_PRECISION* _xdfmica;   // (:,:,:,:,:)
    XS_PRECISION* _xdfmicd;   // (:,:,:,:,:)
    XS_PRECISION* _xdmmics;   // (:,:,:,:,:,:)
    XS_PRECISION* _xdmmica;   // (:,:,:,:,:)
    XS_PRECISION* _xdmmicd;   // (:,:,:,:,:)
    XS_PRECISION* _xddmics;   // (:,:,:,:,:,:)
    XS_PRECISION* _xddmica;   // (:,:,:,:,:)
    XS_PRECISION* _xddmicd;   // (:,:,:,:,:)
    XS_PRECISION* _xdpmics;   // (:,:,:,:,:,:)
    XS_PRECISION* _xdpmica;   // (:,:,:,:,:)
    XS_PRECISION* _xdpmicd;   // (:,:,:,:,:)
    XS_PRECISION* _xdpmicn;   // (:,:,:,:,:)
    XS_PRECISION* _xdfmicn;   // (:,:,:,:,:)
    XS_PRECISION* _xdmmicn;   // (:,:,:,:,:)
    XS_PRECISION* _xddmicn;   // (:,:,:,:,:)

//
// pointers for micro cross sections
//
// xsmicd    transport
// xsmica    absorption
// xsmics    scatter
// xsmicf    fission
// xsmick    kappa-fission
// xsmicn    nue-fission
//
//        => those at intermediate stage
//           and at end of each burnup step(final)
//
// xsmicd0   transport
// xsmica0   absorption
// xsmics0   scatter
// xsmicf0   fission
// xsmick0   kappa-fission
// xsmicn0   nue-fission
//
//        => those at each burnup step
//
//
    XS_PRECISION* _xsmicd; // (:,:,:,:)
    XS_PRECISION* _xsmica; // (:,:,:,:)
    XS_PRECISION* _xsmics; // (:,:,:,:,:)
    XS_PRECISION* _xsmicf; // (:,:,:,:)
    XS_PRECISION* _xsmick; // (:,:,:,:)
    XS_PRECISION* _xsmicn; // (:,:,:,:)
    XS_PRECISION* _xsmic2n; // (:,:)

    XS_PRECISION* _xsmicd0; // (:,:,:,:)
    XS_PRECISION* _xsmica0; // (:,:,:,:)
    XS_PRECISION* _xsmics0; // (:,:,:,:,:)
    XS_PRECISION* _xsmicf0; // (:,:,:,:)
    XS_PRECISION* _xsmick0; // (:,:,:,:)
    XS_PRECISION* _xsmicn0; // (:,:,:,:)

public:
    __host__ CrossSection(const int& ng, const int& nxyz, XS_PRECISION* xsdf, XS_PRECISION* xstf, XS_PRECISION* xsnf, XS_PRECISION* xssf, XS_PRECISION* xschif, XS_PRECISION* xsadf) {
        _ng = ng;
        _nxyz = nxyz;
        _xsnf = xsnf;
        _xsdf = xsdf;
        _xstf = xstf;
        _chif = xschif;
        _xssf = xssf;
        _xsadf = xsadf;
    };

    __host__ CrossSection(const int& ng, const int& mnucl, const int& nxyz) {
        _ng = ng;
        _nxyz = nxyz;
        _xsnf = new XS_PRECISION[_ng * _nxyz]{};
        _xsdf = new XS_PRECISION[_ng*_nxyz]{};
        _xstf = new XS_PRECISION[_ng*_nxyz]{};
        _xskf = new XS_PRECISION[_ng*_nxyz]{};
        _chif = new XS_PRECISION[_ng*_nxyz]{};
        _xssf = new XS_PRECISION[_ng*_ng*_nxyz]{};
        _xsadf = new XS_PRECISION[_ng * _nxyz]{};

        _xsmicd = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:)
        _xsmica = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:)
        _xsmics = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:,:)
        _xsmicf = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:)
        _xsmick = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:)
        _xsmicn = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:)

        _xsmic2n = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:)
        _xsmicd0 = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:)
        _xsmica0 = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:)
        _xsmics0 = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:,:)
        _xsmicf0 = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:)
        _xsmick0 = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:)
        _xsmicn0 = new XS_PRECISION[_ng*mnucl*_nxyz]; // (:,:,:,:)

        _xdfmicd = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xddmicd = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xdpmicd = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xdpmicf = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xdfmicf = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xddmicf = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xdfmica = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xddmica = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xdpmica = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xdpmicn = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xdfmicn = new XS_PRECISION[_ng*mnucl*_nxyz];
        _xddmicn = new XS_PRECISION[_ng*mnucl*_nxyz];

        _xdfmics = new XS_PRECISION[_ng*_ng*mnucl*_nxyz];
        _xddmics = new XS_PRECISION[_ng*_ng*mnucl*_nxyz];
        _xdpmics = new XS_PRECISION[_ng*_ng*mnucl*_nxyz];

        _xdmmicd = new XS_PRECISION[_ng * _nptm *mnucl*_nxyz];
        _xdmmicf = new XS_PRECISION[_ng * _nptm *mnucl*_nxyz];
        _xdmmica = new XS_PRECISION[_ng * _nptm *mnucl*_nxyz];
        _xdmmicn = new XS_PRECISION[_ng * _nptm *mnucl*_nxyz];
        _xdmmics = new XS_PRECISION[_ng  * _ng * _nptm*mnucl*_nxyz];



    };

    __host__ virtual ~CrossSection() {
        delete [] _xsnf;
        delete [] _xsdf;
        delete [] _xstf;
        delete [] _xskf;
        delete [] _chif;
        delete [] _xssf;

        delete [] _xsmicd ;
        delete [] _xsmica ;
        delete [] _xsmics ;
        delete [] _xsmicf ;
        delete [] _xsmick ;
        delete [] _xsmicn ;

        delete [] _xsmic2n ;
        delete [] _xsmicd0 ;
        delete [] _xsmica0 ;
        delete [] _xsmics0 ;
        delete [] _xsmicf0 ;
        delete [] _xsmick0 ;
        delete [] _xsmicn0 ;

        delete [] _xdfmicd ;
        delete [] _xdmmicd ;
        delete [] _xddmicd ;
        delete [] _xdpmicd ;
        delete [] _xdpmicf ;
        delete [] _xdfmicf ;
        delete [] _xdmmicf ;
        delete [] _xddmicf ;
        delete [] _xdfmica ;
        delete [] _xdmmica ;
        delete [] _xddmica ;
        delete [] _xdpmica ;
        delete [] _xdpmicn ;
        delete [] _xdfmicn ;
        delete [] _xdmmicn ;
        delete [] _xddmicn ;

        delete [] _xdfmics ;
        delete [] _xdmmics ;
        delete [] _xddmics ;
        delete [] _xdpmics ;
    }

    __host__ __device__ inline XS_PRECISION& xsnf(const int & ig, const int & l) {return _xsnf[l*_ng+ig];};
    __host__ __device__ inline XS_PRECISION& xsdf(const int & ig, const int & l) {return _xsdf[l*_ng+ig];};
    __host__ __device__ inline XS_PRECISION& xssf(const int & igs, const int & ige, const int & l) {return _xssf[l*_ng*_ng+ige*_ng+igs];};
    __host__ __device__ inline XS_PRECISION& xstf(const int & ig, const int & l) {return _xstf[l*_ng+ig];};
    __host__ __device__ inline XS_PRECISION& xskf(const int & ig, const int & l) {return _xskf[l*_ng+ig];};
    __host__ __device__ inline XS_PRECISION& chif(const int & ig, const int & l) {return _chif[l*_ng+ig];};
    __host__ __device__ inline XS_PRECISION& xsadf(const int& ig, const int& l) { return _xsadf[l * _ng + ig]; };

    __host__ __device__ inline XS_PRECISION& xdfmicd(const int& ig, const int& iiso, const int& l) { return _xdfmicd[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xddmicd(const int& ig, const int& iiso, const int& l) { return _xddmicd[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdpmicd(const int& ig, const int& iiso, const int& l) { return _xdpmicd[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdpmicf(const int& ig, const int& iiso, const int& l) { return _xdpmicf[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdfmicf(const int& ig, const int& iiso, const int& l) { return _xdfmicf[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xddmicf(const int& ig, const int& iiso, const int& l) { return _xddmicf[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdfmica(const int& ig, const int& iiso, const int& l) { return _xdfmica[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xddmica(const int& ig, const int& iiso, const int& l) { return _xddmica[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdpmica(const int& ig, const int& iiso, const int& l) { return _xdpmica[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdpmicn(const int& ig, const int& iiso, const int& l) { return _xdpmicn[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdfmicn(const int& ig, const int& iiso, const int& l) { return _xdfmicn[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xddmicn(const int& ig, const int& iiso, const int& l) { return _xddmicn[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdfmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xdfmics[l * _ng * _ng * _mnucl + iiso*_ng*_ng + ige*_ng + igs]; };
    __host__ __device__ inline XS_PRECISION& xddmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xddmics[l * _ng * _ng * _mnucl + iiso*_ng*_ng + ige*_ng + igs]; };
    __host__ __device__ inline XS_PRECISION& xdpmics(const int& igs, const int& ige, const int& iiso, const int& l) { return _xdpmics[l * _ng * _ng * _mnucl + iiso*_ng*_ng + ige*_ng + igs]; };

    __host__ __device__ inline XS_PRECISION& xdmmicd(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicd[l * _ng * _nptm * _mnucl+ iiso*_ng*_nptm + ip*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdmmicf(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicf[l * _ng * _nptm * _mnucl+ iiso*_ng*_nptm + ip*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdmmica(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmica[l * _ng * _nptm * _mnucl+ iiso*_ng*_nptm + ip*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdmmicn(const int& ig, const int& ip, const int& iiso, const int& l) { return _xdmmicn[l * _ng * _nptm * _mnucl+ iiso*_ng*_nptm + ip*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xdmmics(const int& igs, const int& ige, const int& ip, const int& iiso, const int& l) { return _xdmmics[l * _ng * _ng * _nptm * _mnucl + iiso*_ng*_ng*_nptm + ip *_ng*_ng + ige*_ng + igs]; };

    __host__ __device__ inline XS_PRECISION& xsmic2n(const int& ig, const int& iiso, const int& l) { return _xsmic2n[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmicd (const int& ig, const int& iiso, const int& l) { return _xsmicd [l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmica (const int& ig, const int& iiso, const int& l) { return _xsmica [l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmics (const int& ig, const int& iiso, const int& l) { return _xsmics [l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmicf (const int& ig, const int& iiso, const int& l) { return _xsmicf [l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmick (const int& ig, const int& iiso, const int& l) { return _xsmick [l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmicn (const int& ig, const int& iiso, const int& l) { return _xsmicn [l * _ng * _mnucl + iiso*_ng + ig]; };

    __host__ __device__ inline XS_PRECISION& xsmicd0(const int& ig, const int& iiso, const int& l) { return _xsmicd0[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmica0(const int& ig, const int& iiso, const int& l) { return _xsmica0[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmicf0(const int& ig, const int& iiso, const int& l) { return _xsmicf0[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmick0(const int& ig, const int& iiso, const int& l) { return _xsmick0[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmicn0(const int& ig, const int& iiso, const int& l) { return _xsmicn0[l * _ng * _mnucl + iiso*_ng + ig]; };
    __host__ __device__ inline XS_PRECISION& xsmics0(const int& igs, const int& ige, const int& iiso, const int& l) { return _xsmics0[l * _ng  * _ng * _mnucl + iiso*_ng * _ng + ige*_ng + igs]; };

    __host__ __device__ void dupdxs(const int& l, const float& ppm, const float& tf, const float& tm, const float& dm);
};
