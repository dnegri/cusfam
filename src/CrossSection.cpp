#include "CrossSection.h"

void CrossSection::dupdxs(const int& l, const float& dppm, const float& dtf, const float& dtm, const float& ddm) {

    float dxp[]{dppm, dtf, ddm, dtm};

    for (int inis = 0; inis < NNIS; ++inis) {
        int iiso = ISONIS[inis];

        for (int ig = 0; ig < _ng; ++ig) {
            xsmicd(ig, iiso, l) = xsmicd0(ig, iiso, l)
                                  + xdpmicd(ig, iiso, l) * dxp[0]
                                  + xdfmicd(ig, iiso, l) * dxp[1]
                                  + xddmicd(ig, iiso, l) * dxp[2];
            xsmica(ig, iiso, l) = xsmica0(ig, iiso, l)
                                  + xdpmica(ig, iiso, l) * dxp[0]
                                  + xdfmica(ig, iiso, l) * dxp[1]
                                  + xddmica(ig, iiso, l) * dxp[2];
            for (int igs = 0; igs < _ng; ++igs) {
                xsmics(igs, ig, iiso, l) = xsmics0(igs, ig, iiso, l)
                                           + xdpmics(igs, ig, iiso, l) * dxp[0]
                                           + xdfmics(igs, ig, iiso, l) * dxp[1]
                                           + xddmics(igs, ig, iiso, l) * dxp[2];
            }
        }

        for (int ip = 0; ip < _nptm; ++ip) {
            for (int ig = 0; ig < _ng; ++ig) {
                xsmicd(ig, iiso, l) = xsmicd(ig, iiso, l) + xdmmicd(ig, ip, iiso, l) * dxp[3];
                xsmica(ig, iiso, l) = xsmica0(ig, iiso, l) + xdmmica(ig, ip, iiso, l) * dxp[3];
                for (int igs = 0; igs < _ng; ++igs) {
                    xsmics(igs, ig, iiso, l) = xsmics0(igs, ig, iiso, l) + xdmmics(igs, ig, ip, iiso, l) * dxp[3];
                }
            }
            dxp[3] *= dtm;
        }

    }
}
