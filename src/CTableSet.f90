module CTableSet
    
    implicit none
    private
    
    
    integer, parameter   :: ns = 60
    integer, parameter   :: nv = 12
    integer, parameter   :: nnucl = 40
    integer, parameter   :: LEN_COMPNAME = 12
    integer, parameter   :: LEN_ISOTOPE = 4
    integer, parameter   :: XSSIG_NTYPE = 4
    integer, parameter   :: XSSIG_NUF = 1
    integer, parameter   :: XSSIG_FIS = 2
    integer, parameter   :: XSSIG_CAP = 3
    integer, parameter   :: XSSIG_TRS = 4
    integer, parameter   :: VAR_NTYPE = 4
    integer, parameter   :: VAR_PPM = 1
    integer, parameter   :: VAR_TF = 2
    integer, parameter   :: VAR_TM = 3
    integer, parameter   :: VAR_DM = 4
    integer, parameter   :: VAR_NPOLY = 2
    integer, parameter   :: VAR_NTNP = XSSIG_NTYPE*VAR_NPOLY
    
    integer, parameter   :: NISO = 40
    integer, parameter   :: ng = 2
    integer, parameter   :: ngs = (ng-1)*ng !skip self-scattering 
    
    character(len=4),parameter, dimension(NISO) :: NNUID =   (/'U234', 'U235', 'U236', 'NP37', 'U238', &
                                                               'PU48', 'NP39', 'PU49', 'PU40', 'PU41', &
                                                               'PU42', 'AM43', 'RESI', 'POIS', 'PM47', &
                                                               'PS48', 'PM48', 'PM49', 'SM29', 'I135', &
                                                               'XE45', 'FP.1', 'B-10', 'H2O ', 'STRM', &
                                                               'AM41', 'AM42', 'CM42', 'CM44', 'TH32', &
                                                               'PA33', 'U233', 'MAC ', 'DEL1', 'DEL2', &
                                                               'DEL3', 'TMOD', 'DETE', '   V', 'XSE '/)
    integer, parameter :: ID_U234 = 1
    integer, parameter :: ID_U235 = 2
    integer, parameter :: ID_U236 = 3
    integer, parameter :: ID_NP37 = 4
    integer, parameter :: ID_U238 = 5
    integer, parameter :: ID_PU48 = 6
    integer, parameter :: ID_NP39 = 7
    integer, parameter :: ID_PU49 = 8
    integer, parameter :: ID_PU40 = 9
    integer, parameter :: ID_PU41 = 10
    integer, parameter :: ID_PU42 = 11
    integer, parameter :: ID_AM43 = 12
    integer, parameter :: ID_RESI = 13
    integer, parameter :: ID_POIS = 14
    integer, parameter :: ID_PM47 = 15
    integer, parameter :: ID_PS48 = 16
    integer, parameter :: ID_PM48 = 17
    integer, parameter :: ID_PM49 = 18
    integer, parameter :: ID_SM29 = 19
    integer, parameter :: ID_I135 = 20
    integer, parameter :: ID_XE45 = 21
    integer, parameter :: ID_FP1  = 22
    integer, parameter :: ID_B10  = 23
    integer, parameter :: ID_H2O  = 24
    integer, parameter :: ID_STRM = 25
    integer, parameter :: ID_AM41 = 26
    integer, parameter :: ID_AM42 = 27
    integer, parameter :: ID_CM42 = 28
    integer, parameter :: ID_CM44 = 29
    integer, parameter :: ID_TH32 = 30
    integer, parameter :: ID_PA33 = 31
    integer, parameter :: ID_U233 = 32
    integer, parameter :: ID_MAC  = 33
    integer, parameter :: ID_DEL1 = 34
    integer, parameter :: ID_DEL2 = 35
    integer, parameter :: ID_DEL3 = 36
    integer, parameter :: ID_TMOD = 37
    integer, parameter :: ID_DETE = 38
    integer, parameter :: ID_V    = 39
    integer, parameter :: ID_XSE  = 40

    integer, parameter :: REFL_BOTTOM = 1
    integer, parameter :: REFL_TOP = 2
    integer, parameter :: REFL_CORNER = 3
    integer, parameter :: REFL_EDGE = 4
    integer, parameter :: REFL_NTYPE = 4
                                                               
    character*(*), parameter    :: FORMAT_HEADER = "(A4,A44,I1,I3,I1,1X,I2,I1,1X,I2,3I4)"
    character*(*), parameter    :: FORMAT_VALUE = "(6E12.6)"


    type, public    :: Isotope
        integer                     :: id
        integer                     :: nax, nvar, nvar2, iwab,nskip,nbltab(5)
        real(4)                     :: xsbu(ns), dxsbu(nv)
        real(4)                     :: cappa(ng)
        real(4)                     :: xssig(ns, ng, XSSIG_NTYPE), dxssig(nv, ng, VAR_NTNP, XSSIG_NTYPE)
        real(4)                     :: xsigs(ns, ngs), dxsigs(nv, ngs, VAR_NTNP)
    end type
                                                               
    type, public    :: Reflector
        real(4)                     :: rasigb(4,ng,3), rasigsb(ng,ng,3)
        real(4)                     :: rasigt(4,ng,3), rasigst(ng,ng,3)
        real(4)                     :: rrsig(4,ng,4), rrsigs(2,ng,ng,2)
        real(4)                     :: rfrppm(REFL_NTYPE), rfrtf(REFL_NTYPE), rfrtm(REFL_NTYPE), rfrdm(REFL_NTYPE) ,rfrprs(REFL_NTYPE), rfratm(REFL_NTYPE), b10ap(REFL_NTYPE)
    end type
    
    type, public    :: Composition
        character*(LEN_COMPNAME)    :: name
        type(Isotope)               :: iso(NISO)
        integer                     :: npoly(4)
        real(4)                     :: refvar(4)
        real(4)                     :: refpress
        real(4)                     :: b10ap
        
        integer                     :: npdet(2)
        real(4)                     :: xsend(ns),xsn2n(ns), chi(ns,ng), dpdet(ns), dxsdet(nv,ng,ns), df(ns,ng,3), ddf(ns,ng,3,3)
        
    end type
    
    type, public   :: TableSet
        integer             :: nrefl
        integer             :: ncomp
        character(len=LEN_COMPNAME), pointer    :: compnames(:)
        type(Composition), pointer  :: comps(:)
        type(Reflector)             :: refl
    contains
        procedure init
        procedure readFile
        procedure readReflComp
        procedure readFuelIsotope
        procedure readFuelComp
        procedure skipFuelComp
        procedure readXSOne
        procedure indexOfIsotope
        procedure indexOfComposition
    end type

contains

    subroutine init(this, ncomp, compnames)
        class(TableSet) :: this
        integer         :: ncomp
        character*(LEN_COMPNAME)   :: compnames(ncomp)
        
        integer         :: i
        
        this%ncomp = ncomp
        
        allocate(this%compnames(ncomp))
        allocate(this%comps(ncomp))
        
        do i = 1, ncomp
            this%compnames(i) = compnames(i)
            this%comps(i)%name = compnames(i)
        enddo
        
    end subroutine


    subroutine readFile(this, filename)
        class(TableSet) :: this
        character*(*)   :: filename
        character*(80)  :: header
        character*(LEN_COMPNAME)    :: compname
        type(Composition), pointer   :: comp
        integer         :: ifile, icomp
        integer         :: npoly(4)
        
        ifile = 1002
        open(ifile,file=filename,status='old')
        
        do while(.true.)
            read (ifile,'(a)',end=1,err=2) header
        
            select case(header(1:4))
            case('COMP')
                READ (header, '(5X,A12,39X,4I4)') compname, npoly(1:4)
         
                icomp = this%indexOfComposition(compname)
                
                if(icomp == 0) then
                    call this%skipFuelComp(ifile)
                else
                    comp => this%comps(icomp)
                    comp%name = compname
                    comp%npoly(1:4) = npoly(1:4)
                    call this%readFuelComp(ifile, comp, header)
                endif
                
            case('REFL')
                call this%readReflComp(ifile, this%refl, header)
            end select
        enddo
1       continue
2       continue
    end subroutine

    subroutine readReflComp(this, ifile, refl, header)
        class(TableSet) :: this
        integer         :: ifile
        type(Reflector)     :: refl
        character*(80)  :: header
        character*(4)       :: nd
        character*(44)      :: titl
        integer             :: nax, iwab, nbltab(5), nskip, lin1, lin2, ires
        real(4)             :: dum(7)
        integer             :: ind, i, ib, ie, j, idum, mxra, i1, ig, j1, igs
        
        read(ifile,FORMAT_HEADER) nd,titl,nax
        read(ifile,'(7f12.5)') dum(:) ! rfrppm(irefl), rfrtf(irefl), rfrtm(irefl), rfrdm(irefl) ,rfrprs(irefl), rfratm(irefl), b10ap        
        select case (header(10:12))
        case('BOT')
            mxra = 2
            if (nax == 3) mxra = 3
            do i1=1,mxra
            do ig=1,ng
               read(ifile,FORMAT_VALUE) refl%rasigb(1:3,ig,i1),refl%rasigsb(ig,1:ig-1,i1), refl%rasigsb(ig,ig+1:ng,i1)
            enddo
            enddo
            refl%rfrppm(REFL_BOTTOM) = dum(1)
            refl%rfrtf(REFL_BOTTOM) = dum(2)
            refl%rfrtm(REFL_BOTTOM) = dum(3)
            refl%rfrdm(REFL_BOTTOM) = dum(4)
            refl%rfrprs(REFL_BOTTOM) = dum(5)
            refl%rfratm(REFL_BOTTOM) = dum(6)
            refl%b10ap(REFL_BOTTOM) = dum(7)
        case('TOP')
            mxra = 2
            if (nax == 3) mxra = 3
            do i1=1,mxra
            do ig=1,ng
               read(ifile,FORMAT_VALUE) refl%rasigt(1:3,ig,i1),refl%rasigst(ig,1:ig-1,i1), refl%rasigsb(ig,ig+1:ng,i1)
            enddo
            enddo      
            refl%rfrppm(REFL_TOP) = dum(1)
            refl%rfrtf(REFL_TOP) = dum(2)
            refl%rfrtm(REFL_TOP) = dum(3)
            refl%rfrdm(REFL_TOP) = dum(4)
            refl%rfrprs(REFL_TOP) = dum(5)
            refl%rfratm(REFL_TOP) = dum(6)
            refl%b10ap(REFL_TOP) = dum(7)
        case('COR')
            do i1=1,2
            do ig=1,ng
               read(ifile,FORMAT_VALUE)  refl%rrsig(:,ig,i1)
            enddo
            enddo
        
            do j1=1,1
            do igs=1,ng
               read(ifile,FORMAT_VALUE) refl%rrsigs(j1,igs,1:igs-1,1), refl%rrsigs(j1,igs,igs+1:ng,1)
            enddo
            enddo
            refl%rfrppm(REFL_CORNER) = dum(1)
            refl%rfrtf(REFL_CORNER) = dum(2)
            refl%rfrtm(REFL_CORNER) = dum(3)
            refl%rfrdm(REFL_CORNER) = dum(4)
            refl%rfrprs(REFL_CORNER) = dum(5)
            refl%rfratm(REFL_CORNER) = dum(6)
            refl%b10ap(REFL_CORNER) = dum(7)
        case('EDG')
            do i1=3,4
            do ig=1,ng
               read(ifile,FORMAT_VALUE)  refl%rrsig(:,ig,i1)
            enddo
            enddo
        
            do j1=1,1
            do igs=1,ng
               read(ifile,FORMAT_VALUE) refl%rrsigs(j1,igs,1:igs-1,1), refl%rrsigs(j1,igs,igs+1:ng,1)
            enddo
            enddo
            refl%rfrppm(REFL_EDGE) = dum(1)
            refl%rfrtf(REFL_EDGE) = dum(2)
            refl%rfrtm(REFL_EDGE) = dum(3)
            refl%rfrdm(REFL_EDGE) = dum(4)
            refl%rfrprs(REFL_EDGE) = dum(5)
            refl%rfratm(REFL_EDGE) = dum(6)
            refl%b10ap(REFL_EDGE) = dum(7)
        end select
    
    end subroutine
    
    subroutine skipFuelComp(this, ifile)
        class(TableSet) :: this
        integer         :: ifile
        character*(80)   :: header

        call skip(ifile, 1)
        
        do while(.true.)
            READ (ifile,'(a)', end=1) header
            
            if(header(1:4) == 'COMP' .or. header(1:4) == 'REFL') then
                backspace(ifile)
                exit
            endif
        enddo
1       continue        
    end subroutine
    
    subroutine readFuelComp(this, ifile, comp, header)
        class(TableSet) :: this
        integer         :: ifile
        type(Composition)   :: comp
        character*(*)   :: header
        integer             :: idiso
        
        READ (ifile,'(6F12.6)')  comp%refvar(1:4), comp%refpress, comp%b10ap
        
        if (comp%b10ap == 0) comp%b10ap = 19.8

        !b10wp1  = b10abnd*b10aw / (b10abnd*b10aw + (100.-b10abnd)*b11aw)
        !ppmcorr = b10wp1*100.0 / b10wp
        !refppm(ixset) = ppmcorr * refppm(ixset)
    
        do while(.true.)
            READ (ifile,'(a)', end=1) header
            
            if(header(1:4) == 'COMP' .or. header(1:4) == 'REFL') then
                backspace(ifile)
                exit
            endif
            
            idiso = this%indexOfIsotope(header(1:4))
            comp%iso(idiso).id = idiso
            call this.readFuelIsotope(ifile, comp, comp%iso(idiso), header)
        enddo
        
1       continue
        
    end subroutine
    
    subroutine readFuelIsotope(this, ifile, comp, iso, header)
        class(TableSet)     :: this
        integer             :: ifile
        type(Composition)   :: comp
        type(Isotope)       :: iso
        character*(*)       :: header
    
        logical             :: n2n
        character*(4)       :: nd
        character*(44)      :: titl
        integer             :: nax, iwab, nbltab(5), nskip, lin1, lin2, ires
        real                :: ppm, tfuel, tmod, dm, pressxs, void
        integer             :: ind, i, ib, ie, j, idum, ixs, mok, idxdrv, ivar, ngnp, idf, ig, ip
        
        if(header(5:7) == 'N2N') n2n = .true.
        
        read(header,FORMAT_HEADER) nd,titl,nax,iso%nvar,iwab,nbltab(1),nskip, (nbltab(i),i=2,5)
        
        call skip(ifile, nskip)
        
        if (iwab > 0) then
            read(ifile,*) idum,idum,iso%nvar2
            call skip(ifile,1)
        endif
        
        if(iso%nvar<0) iso%nvar=-iso%nvar

        read(ifile,FORMAT_VALUE) ppm,tfuel,tmod,dm,pressxs,void

        lin1 = iso%nvar/6
        ires = mod(iso%nvar,6)
        if(ires>0) lin1=lin1+1
        
        if(iso%id == ID_DETE .or. iso%id == ID_V) then
            comp%npdet(1) = iso%nvar
            comp%npdet(2) = nax
            call this%readXSOne(ifile, iso%nvar, lin1, 1, ns, comp%dpdet)
            
            lin2=iso%nvar2/6
            ires=mod(iso%nvar2,6)
            if(ires.gt.0) lin2=lin2+1
            
            call skip(ifile, 1)
            
            do i=1,iso%nvar
                call this%readXSOne(ifile, iso%nvar2, lin2, ng, nv, comp%dxsdet(:,:,i))
            enddo
            
            return
        endif

        call this%readXSOne(ifile, iso%nvar, lin1, 1, ns, iso%xsbu)

        if (iwab.gt.0) then
            lin2=iso%nvar2/6
            ires=mod(iso%nvar2,6)
            if(ires.gt.0) lin2=lin2+1
            
            call this%readXSOne(ifile, iso%nvar2, lin2, 1, nv, iso%dxsbu)
        endif

        if(iso%id == ID_XSE) then
            call this%readXSOne(ifile, iso%nvar, lin1, 1, ns, comp%xsend)
        endif
        
        
        if(nbltab(1)>0) read(ifile,FORMAT_VALUE) (iso%cappa(i),i=1,ng)
        
        if(iso%id == ID_U238 .and. n2n) then
            call this%readXSOne(ifile, iso%nvar, lin1, 1, ns, comp%xsn2n)
        endif
        if(iso%id == ID_MAC) then
            call this%readXSOne(ifile, iso%nvar, lin1, ng, ns, comp%chi)
        endif
        

!        write values on xscomp
!        ixset : no. of composition
!        ind   : sequential no. of isotope
!
         ! the order of xsec types :
         ! nu-fission(1), fission(2), capture(3), transport(4,skipped)
         do ixs=1,XSSIG_NTYPE
            if(nbltab(ixs)==0) cycle

            call this%readXSOne(ifile, iso%nvar, lin1, ng, ns, iso%xssig(:,:,ixs))

            ! 1111 --> from right, 1st;ppm, 2nd;tf, 3rd;tm, 4th;dm
            if(nbltab(ixs)==16) cycle

            mok=nbltab(ixs)
            idxdrv = 0
            do ivar=1,VAR_NTYPE
               if(mod(mok,2)==1) then
                  
                  do ig=1,ng
                  do ip = 0, comp%npoly(ivar)
                    call this%readXSOne(ifile, iso%nvar2, lin2, 1, nv, iso%dxssig(:,ig,idxdrv+ip+1,ixs))
                  enddo
                  enddo
                  
                  if(comp%npoly(ivar) == 0) then
                    iso%dxssig(:,:,idxdrv+1,ixs) = iso%dxssig(:,:,idxdrv,ixs)
                    iso%dxssig(:,:,idxdrv,ixs) = 0.0
                  endif
                  ngnp = 1+comp%npoly(ivar)
                  idxdrv = idxdrv + ngnp
               endif
               mok=mok*0.5
               if(mok<=0) exit
            enddo
         enddo ! ixs

         ! scattering xs
         ixs=5
         if(nbltab(ixs)>0) then
           call this%readXSOne(ifile, iso%nvar, lin1, ngs, ns, iso%xsigs(:,:))

            if(nbltab(ixs) /= 16) then
               mok=nbltab(ixs)
               idxdrv = 1
               ! variation type : ppm, tf, tm, dm
               do ivar=1,VAR_NTYPE
                  if(mod(mok,2)==1) then

                     do ig=1,ngs
                     do ip = 0, comp%npoly(ivar)
                        call this%readXSOne(ifile, iso%nvar2, lin2, 1, nv, iso%dxsigs(:,ngs,idxdrv+ip+1))
                     enddo
                     enddo

                     if(comp%npoly(ivar) == 0) then
                        iso%dxsigs(:,:,idxdrv+1) = iso%dxsigs(:,:,idxdrv)
                        iso%dxsigs(:,:,idxdrv) = 0.0
                     endif
                     idxdrv = idxdrv + 1 + comp%npoly(ivar)
                  endif
                  mok=mok*0.5
                  if(mok <= 0) exit
               enddo
            endif
         endif  ! nbltab(i)>0
         
        if(iso%id == ID_MAC) then
            do idf=1,3    ! adf, cdf, mdf
                call this%readXSOne(ifile, iso%nvar, lin1, ng, ns, comp%df(:,:,idf))
            enddo            
        endif
        
        if(iso%id == ID_DEL1 .or. iso%id == ID_DEL2 .or. iso%id == ID_DEL3 ) then
            do idf=1,3    ! adf, cdf, mdf
                call this%readXSOne(ifile, iso%nvar, lin1, ng, ns, comp%ddf(:,:,idf,ID_DEL1-ID_MAC))
            enddo            
        endif
         
    end subroutine

    subroutine readXSOne(this, ifile, nvar, lin1, ngrp, nbu, xs)
        class(TableSet)             :: this
        integer                     :: ifile, nvar, lin1, ngrp, nbu
        integer                     :: ig, i1, ib, ie, j
        real(4)                     :: xs(nbu, ngrp)

        do ig=1,ngrp
            do i1=1,lin1
                ib=(i1-1)*6+1
                ie=ib+5
                if(ie>nvar) ie=nvar
                read(ifile,FORMAT_VALUE) (xs(j,ig),j=ib,ie)
            enddo
        enddo
    end subroutine
    
    function indexOfIsotope(this, nmiso) result(iiso)
        class(TableSet)     :: this
        integer             :: iiso
        character*(4)       :: nmiso
        
        do iiso = 1, NISO
            if(nmiso == NNUID(iiso)) exit
        enddo

    end function

    subroutine skip(ifile,nline)
        integer     :: ifile
        integer     :: nline
        integer     :: i

        do i=1,nline
            read(ifile,'(a)') 
        enddo
    end subroutine
    
    function indexOfComposition(this, compname) result(icomp)
        class(TableSet)     :: this
        character*(*)       :: compname
        integer             :: icomp, i
        
        icomp = 0
        do i=1,this%ncomp
            if(this%compnames(i) == compname) then
                icomp = i
                exit
            endif
        enddo
    end function
    
    subroutine calculate(this, burn, xsmicd, xsmica, xsmicn, xsmicf, xsmick, xsmics, xsmic2n, xehfp)
        class(Isotope)      :: this
        real(4)             :: burn
        real(4)             :: xsmicd(ng,NISO), xsmica(ng,NISO), xsmicn(ng,NISO), xsmicf(ng,NISO), xsmick(ng,NISO), xsmics(ng,ng,NISO), xsmic2n(ng), xehfp(ng)
        integer             :: klo
        real(4)             :: af(3)
        
        call quad1(this%nvar, this%xsbu(:), burn, klo, af)

        xsmic2n(1) = af(1) * this%xsn2n(klo) + af(2) * this%xsn2n(klo+1) + af(3) * this%xsn2n(klo+2)
        xehfp = af(1) * this%xsend(klo) + af(2) * this%xsend(klo+1) + af(3) * this%xsend(klo+2)
        
        do iso = 1, NISO
            this%calculate(klo, af, xsmicd(:,iso), xsmica(:,iso), xsmicn(:,iso), xsmicf(:,iso), xsmick(:,iso), xsmics(:,iso))
        enddo
        
    end subroutine
    
    subroutine calculateVariation(this, burn,   xdpmicn, xdfmicn, xdmmicn, xddmicn, &
                                                xdpmicf, xdfmicf, xdmmicf, xddmicf, &
                                                xdpmica, xdfmica, xdmmica, xddmica, &
                                                xdpmicd, xdfmicd, xdmmicd, xddmicd )
        class(Isotope)      :: this
        real(4)             :: burn
        real(4)             ::  xdpmicn(ng,NISO), xdfmicn(ng,NISO), xdmmicn(ng,3,NISO), xddmicn(ng,NISO), &
                                xdpmicf(ng,NISO), xdfmicf(ng,NISO), xdmmicf(ng,3,NISO), xddmicf(ng,NISO), &
                                xdpmica(ng,NISO), xdfmica(ng,NISO), xdmmica(ng,3,NISO), xddmica(ng,NISO), &
                                xdpmicd(ng,NISO), xdfmicd(ng,NISO), xdmmicd(ng,3,NISO), xddmicd(ng,NISO)
        integer             :: klo
        real(4)             :: af(3)
        
        call quad1(this%nvar2, this%dxsbu(:), burn, klo, af)

        do ig=1,ng
            this%calculateVariation(klo, af, 1, ig, xdpmicn(:,iso), xdfmicn(:,iso), xdmmicn(:,:,iso), xddmicn(:,iso))
            this%calculateVariation(klo, af, 2, ig, xdpmicf(:,iso), xdfmicf(:,iso), xdmmicf(:,:,iso), xddmicf(:,iso))
            this%calculateVariation(klo, af, 3, ig, xdpmica(:,iso), xdfmica(:,iso), xdmmica(:,:,iso), xddmica(:,iso))
            xdpmica = xdpmica + xdpmicf
            xdfmica = xdfmica + xdfmicf
            xdmmica = xdmmica + xdmmicf
            xddmica = xddmica + xddmicf
            this%calculateVariation(klo, af, 3, ig, xdpmicd(:,iso), xdfmicd(:,iso), xdmmicd(:,:,iso), xddmicd(:,iso))
        enddo
        
    end subroutine    
    
    
    
    subroutine calculate(this, klo, af, xsmicd, xsmica, xsmicn, xsmicf, xsmick, xsmics)
        class(Isotope)     :: this
        
        ixs = 1
        do ig=1,ng
            xsmicn(ig) = af(1) * this%xssig(klo,ig,ixs) + af(2) * this%xssig(klo+1,ig,ixs) + af(3) * this%xssig(klo+2,ig,ixs)
        enddo
        ixs = 2
        do ig=1,ng
            xsmicf(ig) = af(1) * this%xssig(klo,ig,ixs) + af(2) * this%xssig(klo+1,ig,ixs) + af(3) * this%xssig(klo+2,ig,ixs)
            xsmick(ig) = xsmicf(ig)*cappa(ig)
        enddo
        ixs = 3
        do ig=1,ng
            xsmica(ig) = af(1) * this%xssig(klo,ig,ixs) + af(2) * this%xssig(klo+1,ig,ixs) + af(3) * this%xssig(klo+2,ig,ixs)
            xsmica(ig) = xsmica(ig) + xsmicf(ig)
        enddo
        ixs = 4
        do ig=1,ng
            xsmicd(ig) = af(1) * this%xssig(klo,ig,ixs) + af(2) * this%xssig(klo+1,ig,ixs) + af(3) * this%xssig(klo+2,ig,ixs)
        enddo
        
        igse = 0
        do igs=1,ng
        do ige=1,ng
            if(igs.eq.ige) cycle
            igse = igse + 1
            xsmics(igs,ige) = af(1) * this%xsigs(klo,igse) + af(2) * this%xsigs(klo+1,igse) + af(3) * this%xsigs(klo+2,igse)
        enddo
        enddo
        
      enddo         
      
    subroutine calculateVariation(this, klo, af, ixs, ig, xdpmic, xdfmic, xdmmic, xddmic)
        class(Isotope)     :: this
        
        xdpmic(ig) = af(1) * this%dxssig(klo,ig,2,ixs) + af(2) * this%xssig(klo+1,ig,2,ixs) + af(3) * this%xssig(klo+2,ig,2,ixs)        
        xdfmic(ig) = af(1) * this%dxssig(klo,ig,4,ixs) + af(2) * this%xssig(klo+1,ig,4,ixs) + af(3) * this%xssig(klo+2,ig,4,ixs)
        do ip=1,3
            xdmmic(ig,ip) = af(1) * this%dxssig(klo,ig,5+ip,ixs) + af(2) * this%xssig(klo+1,ig,5+ip,ixs) + af(3) * this%xssig(klo+2,ig,5+ip,ixs)
        enddo
        xddmic(ig) = af(1) * this%dxssig(klo,ig,10,ixs) + af(2) * this%xssig(klo+1,ig,10,ixs) + af(3) * this%xssig(klo+2,ig,10,ixs)
    end subroutine
end module
