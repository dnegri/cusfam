!**********************************************************************!
!                                                                      !
! KEPCO Nuclear Fuel CONFIDENTIAL                                      !
! __________________                                                   !
!                                                                      !
!  [2007] - [2013] KEPCO Nuclear Fuel Incorporated                     !
!  All Rights Reserved.                                                !
!                                                                      !
! NOTICE:  All information contained herein is, and remains the        ! 
!          property of KEPCO Nuclear Fuel Incorporated and its         !
!          suppliers, if any. The intellectual and technical           !
!          concepts contained herein are proprietary to KEPCO          !
!          Nuclear Fuel Incorporated and protected by trade            !
!          secret or copyright law.                                    !
!          Dissemination of this information or reproduction of        !
!          this material is strictly forbidden unless prior            !
!          written permission is obtained from KEPCO Nuclear           !
!          Fuel Incorporated.                                          !
!**********************************************************************!

      module sfamintg
      use, intrinsic :: iso_c_binding
      use const
      use allocs
      implicit none
! intg. with geom
      real,pointer,dimension(:,:,:)     :: hmesh
      
! intg. with xsec
      real,pointer,dimension(:,:,:)     :: xsaf, &
                                           xstrf
     
      real,pointer,dimension(:,:,:,:)   :: xscdf
     
      real,pointer,dimension(:,:,:,:)   :: xbeta
      real,pointer,dimension(:,:,:,:,:) :: jnet
      
      real,pointer,dimension(:,:,:)     :: xslmbdk, &
                                           xsbetak, &
                                           xsrvelof
      integer,pointer                      :: nxsf(:), nxef(:)
      real,pointer,dimension(:,:,:,:)   :: xschifd
      real,pointer                      :: xsadf(:,:,:)
      
      integer,pointer                   :: ibegsf(:,:,:) , iendsf(:,:,:)
        integer, pointer                :: nodel(:,:), nxs(:), nxe(:), nys(:), nye(:)
      
      interface
      end interface
!      
      contains     
      subroutine initsfamgeom(  ng,     ndim,   part,           &
                                nx,     ny,     nz,     nxy,     &
                                kbc, kec, nxsl, nxel, nysl, nyel,  &
                                ijtol, hxyz, vol, albedo)  bind(c, name="initsfamgeom")
        use geom, only : setgeom
        use sfam,    only : mallocsfam

        integer                         :: ng, ndim
        integer                         :: nx, ny, nz, nxy
        real                            :: part
        integer, target, intent(in)     :: nxsl(ny), nxel(ny),nysl(nx), nyel(nx)
        integer, target, intent(in)     :: ijtol(nx,ny)
        real, target, intent(in)        :: hxyz(3,nxy,nz)
        real, target, intent(in)        :: vol(nxy,nz)
        real                            :: albedo(2, NDIRMAX)
        real                            :: volcor, volall
        integer                         :: kbc, kec

        character*4                     :: symopt   
        integer                         :: isymang,isymloc
        integer                         :: l,k,m,i,j,i0, j0, j1, jj, i1,ii,irot
        integer                         :: jfa, ifa, m2dfa, m2d, li, ja, ia, la

! hmesh
        call dmalloc0(hmesh,0,ndirmax,1,nxy,1,nz)
        call dmalloc(nodel,nx,ny)
        do k=1,nz
        do l=1,nxy
            hmesh(0,l,k)=hxyz(XDIR,l,k)
            hmesh(XDIR,l,k)=hxyz(XDIR,l,k)
            hmesh(YDIR,l,k)=hxyz(YDIR,l,k)
            hmesh(ZDIR,l,k)=hxyz(ZDIR,l,k)
        enddo
        enddo
        call dmalloc(nxs,ny)
        call dmalloc(nxe,ny)
        do j=1,ny
            nxs(j) = nxsl(j)+1
            nxe(j) = nxel(j)
            do i=nxs(j), nxe(j)
                nodel(i,j) = ijtol(i,j)+1
            enddo
        enddo
        
        call dmalloc(nys,nx)
        call dmalloc(nye,nx)
        do i=1,nx
            nys(i) = nysl(i)+1
            nye(i) = nyel(i)
        enddo
        
! symopt
        irot = 1
        isymang = 360*part
        if(irot.eq.0) then
            symopt = 'REFL'
        elseif(irot.eq.1) then
            symopt = 'ROT'        
        elseif(irot.eq.2) then
            symopt = 'ROT'        
        endif
               
        isymloc = 4
        
        call dmalloc(nxsf, ny)
        call dmalloc(nxef, ny)
        nxsf = nxs
        nxef = nxe
        
        call setgeom(ng,nx,ny,nz,nxy,ndim,symopt,isymang,isymloc,   &
                     nxs,nxe,nys,nye,nxsf,nxef,                     &
                     kbc,kec,nodel, hmesh, vol, volall, volcor, albedo)
                     
                     
        call mallocsfam(ng,nxy, nz, ndim)
        return
        
      end subroutine
      
      subroutine initsfamxsec(  ng      , ng2s    , ndim    , nxy     , nz      , &
                                xschif  , xsdf , xstf , xsnf ,  &
                                xskf , xssf )   bind(c, name="initsfamxsec")
     
        use xsec, only : setxsec

        integer                         :: ng, ng2s, ndim,nxy, nz
        real, target, intent(in),dimension(ng,nxy,nz) :: xschif , xsdf, xstf , xskf, xsnf
        real,target, intent(in),dimension(ng,ng,nxy,nz) :: xssf
        integer                         :: l,k
        
        
        call dmalloc(xsaf,ng,nxy,nz)
        call dmalloc(xscdf,4,ng,nxy,nz)
        call dmalloc(xbeta,ng,nxy,nz,ndirmax)
        
        call dmalloc(ibegsf, ng,nxy,nz)
        call dmalloc(iendsf, ng,nxy,nz)
        call dmalloc(xsadf, ng,nxy,nz)
        ibegsf = 1
        iendsf = 2
        xsadf  = 1.0
        xscdf = 1.0
        
        call setxsec(ng,        ng2s,      nxy,    nz,          &
                     xstf,      xsaf,      xsdf,   xsnf,        &
                     xskf,      xschif,     xbeta,              &
                     xsadf,     xscdf,      xssf, ibegsf, iendsf)
      end subroutine
      
      subroutine updsfamxsec()  bind(c, name="updsfamxsec")
        use xsec,   only : xstf,xsaf,xsdf,xbeta,xssf,xskpf,xsnff,xsadf
        use geom,   only : ng,nxy,nz
        integer             :: l,k,m,idir
        
        do k=1,nz
        do l=1,nxy
        do m=1,ng
            do idir=1,ndirmax
                xbeta(m,l,k,idir)=xsdf(m,l,k)/hmesh(idir,l,k)
            enddo
        enddo
        enddo
        enddo
      end subroutine
      
      subroutine updpower(power0, power)
        use const
        use geom,   only : kfbeg,kfend,nx,ny,nxsf,nxef,nodel,volnode,volfuel
        use xsec,   only : xskp
        use sfam,   only : phi
        
        real                :: power0, power
        
        integer             :: i,j,l,k,m
        real                :: pownode,totpow,avgpow
        
        do k=kfbeg,kfend
          do j=1,ny
            if(nxsf(j).eq.0) cycle
            do i=nxsf(j),nxef(j)
              l=nodel(i,j)
              pownode=0.
              do m=1,ng2
                pownode=pownode+xskp(m,l,k)*phi(m,l,k)
              enddo
              pownode=pownode*volnode(l,k)
              totpow=totpow+pownode
            enddo
          enddo
        enddo      

        avgpow=totpow/volfuel
        power=avgpow*power0      
      end subroutine

      subroutine runss(iternew, ifnodal, epsflx, flux, eigvl, reigvl, errflx) bind(c, name="runss")
        use geom
        use sfam,     only :        resetsfam       &
                                , runsteady       &
                                , eigv        &
                                , reigv       &
                                , phif

        use xsec,     only  :       xsadf
        
        use nodal,    only :        trlcff0
     
!       arguments  
        logical           ::        iternew, ifnodal
        
        real            :: epsflx, errflx
        
        real             :: flux(ng,nxy,nz)
        real              :: eigvl, reigvl
     
!       local variables
        integer, save     ::                &
                                  noutbegmg=0     &
                                , ninbegmg=0      &
                                , noutbeg2g=0     &
                                , ninbeg2g=0
     
        integer           ::        modirot     &
                                , modloc      &
                                , idir, lr        &
                                , k, j, i     &
                                , ig, m2d
     
        real              ::        erreig

        phif(:,1:,1:)= flux 
        
!       update local XSECs needed for calculating steady-state 
        call updsfamxsec()

!       update variables related to XSECs.
        call resetsfam()      

!       calculate steady-state with updated XSECs.
        if(iternew) then
            noutbegmg = 0
            noutbeg2g = 0
            ninbegmg  = 0
            ninbeg2g  = 0
        endif
        call runsteady( .true.,  ifnodal,     &
                       noutbegmg,  ninbegmg,     &
                       noutbeg2g,  ninbeg2g,     &
                       epsflx, erreig  ,         &
                       errflx) ! run nodal
     
        eigvl   = eigv
        reigvl  = reigv
        flux = phif(:,1:,1:)

      end subroutine
            
      end module
