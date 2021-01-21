//
// Created by JOO IL YOON on 2021/01/21.
//

#include "CMFDCPU.h"

#define flux(ig, l)   (flux[(l)*_g.ng()+ig])
#define aflux(ig, l)   (aflux[(l)*_g.ng()+ig])

CMFDCPU::CMFDCPU(Geometry &g, CrossSection &x) : CMFD(g, x) {

}
CMFDCPU::~CMFDCPU() {

}

void CMFDCPU::upddtil() {
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddtil(ls);
    }
}
void CMFDCPU::upddhat() {
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddtil(ls);
    }

}

void CMFDCPU::setls() {
    for (int l = 0; l < _g.nxyz(); ++l) {
        CMFD::setls(l);
    }
}

void CMFDCPU::axb(float * flux, double* aflux) {
    for (int l = 0; l < _g.nxyz(); ++l) {
        for (int ig = 0; ig < _g.ng(); ++ig) {
            aflux(ig, l) = am(0,ig,l) * flux(0, l) + am(1, ig, l) * flux(1, l);

            for (int idir = 0; idir < NDIRMAX; ++idir) {
                for (int lr = 0; lr < LR; ++lr) {
                    int ln = _g.neib(lr,idir,l);
                    if(ln != -1)
                        aflux(ig, l) += cc(lr,idir,ig,l) * flux(ig,ln);
                }
            }
        }
    }
}


double CMFDCPU::residual(const double& reigv, const double& reigvs, float* flux) {

    double reigvdel=reigv-reigvs;

//    axb(phi,aphi);
    double r = 0.0;
    double psi2 = 0.0;

    for (int l = 0; l < _g.nxyz(); ++l) {
        double fs=psi(l)*reigvdel;

        for (int ig = 0; ig < _g.ng(); ++ig) {
            double ab = am(0,ig,l) * flux(0, l) + am(1, ig, l) * flux(1, l);

            for (int idir = 0; idir < NDIRMAX; ++idir) {
                for (int lr = 0; lr < LR; ++lr) {
                    int ln = _g.neib(lr,idir,l);
                    if(ln != -1)
                        ab += cc(lr,idir,ig,l) * flux(ig,ln);
                }
            }

            double err = _x.chif(ig,l)*fs-ab;
            r += err*err;

            double ps = _x.chif(ig,l)*psi(l);
            psi2 += ps*ps;
        }
    }

    return sqrt(r/psi2);
}

void CMFDCPU::drive(double& eigv, float* flux, float& errl2) {

    int icy     = 0;
    int icmfd   = 0;
    double reigv = 1./eigv;
    double reigvsdel=0, reigvsd=0;
    double resid0;

    for (int iout = 0; iout < _ncmfd; ++iout) {
        icy=icy+1;
        icmfd=icmfd+1;
        double eigvd = eigv;
        double reigvdel=reigv -_reigvs;
        for (int l = 0; l < _g.nxyz(); ++l) {
            double fs=psi(l)*reigvdel;
            for (int ig = 0; ig < _g.ng(); ++ig) {
                src(ig,l)=_x.chif(ig,l)*fs;
            }
        }

        //solve linear system A*phi = src
        // update flux

        //wielandt shift

        reigvsdel=_reigvs - reigvsd;
        reigvdel=reigv-_reigvs;

        for (int l = 0; l < _g.nxyz(); ++l) {
            am(0,0,l)=am(0,0,l)-_x.xsnf(0,l)*_g.vol(l)*reigvsdel*_x.chif(0,l);
            am(1,1,l)=am(1,1,l)-_x.xsnf(1,l)*_g.vol(l)*reigvsdel*_x.chif(1,l);
            am(1,0,l) = -_x.xssf(1,0,l)*_g.vol(l) - af(1,l)*_reigvs;
            am(0,1,l) = -_x.xssf(0,1,l)*_g.vol(l) - af(1,l)*_reigvs;

        }
        reigvsd=_reigvs;

        double resi = residual(reigv,_reigvs, flux);

        if(icmfd == 0) resid0 = resi;
        double relresid = resi/resid0;

        int negative=0;
        for (int l = 0; l < _g.nxyz(); ++l) {
            for (int ig = 0; ig < _g.ng(); ++ig) {
                if(flux(ig,l) < 0) {
                    ++negative;
                }
            }
        }

        double erreig=abs(eigv-eigvd)

        if(errl2.lt._epsl2)   exit

    }

    //wiel(icy, phi, psi, eigv, reigv, errl2, errlinf);




    if(icmfd.eq.1) resid0 = resid
    relresid = resid/resid0

    negative=0
    do k=1,nz
    do l=1,nxy
    do m=1,ng2
    if(phi(m,l) .le. 0) then
            negative=negative+1
    endif
            enddo
    enddo
            enddo

    erreig=abs(eigv-eigvd)

    write(mesg,100) ibeg+icmfd, eigv, erreig, errl2, resid, relresid, ninner, r2
    call message(TRUE,TRUE,mesg)

    if(ng.eq.ng2 .and. negative .ne. 0 .and. negative .ne. nxy*nz*ng) then
            iout=iout-1
    write(mesg,'(a,i6,"/",i6)') 'NEGATIVE FLUX : ', negative, nxy*nz*ng
    call message(true,true,mesg)
    endif

    if(errl2.lt.epsl2)   exit
                enddo

        ! fixing negative flux.
    if(negative .ge. nxy*nz*ng*0.9) then
        write(mesg,'(a,i6,"/",i6)') 'NEGATIVE FIXUP : ', negative, nxy*nz*ng
    call message(true,true,mesg)
    do k=1,nz
    do l=1,nxy
    phi(:,l)=-1*phi(:,l)
    enddo
            enddo
    negative = nxy*nz*ng - negative
    endif

    if(negative.gt.0 .and. negative .lt. nxy*nz*ng*0.1) then
    write(mesg,'(a,i6,"/",i6)') 'NEGATIVE TO ZERO : ', negative, nxy*nz*ng
    call message(true,true,mesg)

    do k=1,nz
    do l=1,nxy
    do m=1,ng2
    if(phi(m,l) .lt. 0) phi(m,l) = 1.E-30
    enddo
            enddo
    enddo
            endif

    if(negative.ne.0) then
    do k=1,nz
    do l=1,nxy
    psi(l) = sum(xsnf(:,l)*phi(:,l))*volnode(l)
    enddo
            enddo
    endif

            iout=min(iout,ncmfd)
    ibeg=ibeg+iout

}


