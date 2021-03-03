module CTableSet
    use CIsotope
    use CReflector
    use CComposition
    implicit none
    private
                                                               
    character*(*), parameter    :: FORMAT_HEADER = "(A4,A44,I1,I3,I1,1X,I2,I1,1X,I2,3I4)"
    character*(*), parameter    :: FORMAT_VALUE = "(6E12.6)"

                                                               
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
        
        read(header,FORMAT_HEADER) nd,titl,nax,comp%nvar,iwab,nbltab(1),nskip, (nbltab(i),i=2,5)
        
        call skip(ifile, nskip)
        
        if (iwab > 0) then
            read(ifile,*) idum,idum,comp%nvar2
            call skip(ifile,1)
        endif
        
        if(comp%nvar<0) comp%nvar=-comp%nvar

        read(ifile,FORMAT_VALUE) ppm,tfuel,tmod,dm,pressxs,void

        lin1 = comp%nvar/6
        ires = mod(comp%nvar,6)
        if(ires>0) lin1=lin1+1
        
        if(iso%id == ID_DETE .or. iso%id == ID_V) then
            comp%npdet(1) = comp%nvar
            comp%npdet(2) = nax
            call this%readXSOne(ifile, comp%nvar, lin1, 1, ns, comp%dpdet)
            
            lin2=comp%nvar2/6
            ires=mod(comp%nvar2,6)
            if(ires.gt.0) lin2=lin2+1
            
            call skip(ifile, 1)
            
            do i=1,comp%nvar
                call this%readXSOne(ifile, comp%nvar2, lin2, ng, nv, comp%dxsdet(:,:,i))
            enddo
            
            return
        endif

        call this%readXSOne(ifile, comp%nvar, lin1, 1, ns, comp%xsbu)

        if (iwab.gt.0) then
            lin2=comp%nvar2/6
            ires=mod(comp%nvar2,6)
            if(ires.gt.0) lin2=lin2+1
            
            call this%readXSOne(ifile, comp%nvar2, lin2, 1, nv, comp%dxsbu)
        endif

        if(iso%id == ID_XSE) then
            call this%readXSOne(ifile, comp%nvar, lin1, 1, ns, comp%xsend)
        endif
        
        
        if(nbltab(1)>0) read(ifile,FORMAT_VALUE) (iso%cappa(i),i=1,ng)
        
        if(iso%id == ID_U238 .and. n2n) then
            call this%readXSOne(ifile, comp%nvar, lin1, 1, ns, comp%xsn2n)
        endif
        if(iso%id == ID_MAC) then
            call this%readXSOne(ifile, comp%nvar, lin1, ng, ns, comp%chi)
        endif
        

!        write values on xscomp
!        ixset : no. of composition
!        ind   : sequential no. of isotope
!
         ! the order of xsec types :
         ! nu-fission(1), fission(2), capture(3), transport(4,skipped)
         do ixs=1,XSSIG_NTYPE
            if(nbltab(ixs)==0) cycle

            call this%readXSOne(ifile, comp%nvar, lin1, ng, ns, iso%xssig(:,:,ixs))

            ! 1111 --> from right, 1st;ppm, 2nd;tf, 3rd;tm, 4th;dm
            if(nbltab(ixs)==16) cycle

            mok=nbltab(ixs)
            idxdrv = 0
            do ivar=1,VAR_NTYPE
               if(mod(mok,2)==1) then
                  
                  do ig=1,ng
                  do ip = 0, comp%npoly(ivar)
                    call this%readXSOne(ifile, comp%nvar2, lin2, 1, nv, iso%dxssig(:,ig,idxdrv+ip+1,ixs))
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
           call this%readXSOne(ifile, comp%nvar, lin1, ngs, ns, iso%xsigs(:,:))

            if(nbltab(ixs) /= 16) then
               mok=nbltab(ixs)
               idxdrv = 1
               ! variation type : ppm, tf, tm, dm
               do ivar=1,VAR_NTYPE
                  if(mod(mok,2)==1) then

                     do ig=1,ngs
                     do ip = 0, comp%npoly(ivar)
                        call this%readXSOne(ifile, comp%nvar2, lin2, 1, nv, iso%dxsigs(:,ngs,idxdrv+ip+1))
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
                call this%readXSOne(ifile, comp%nvar, lin1, ng, ns, comp%df(:,:,idf))
            enddo            
        endif
        
        if(iso%id == ID_DEL1 .or. iso%id == ID_DEL2 .or. iso%id == ID_DEL3 ) then
            do idf=1,3    ! adf, cdf, mdf
                call this%readXSOne(ifile, comp%nvar, lin1, ng, ns, comp%ddf(:,:,idf,ID_DEL1-ID_MAC))
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
    
    subroutine calculateReference(this, icomp, burn, xsmicd, xsmica, xsmicn, xsmicf, xsmick, xsmics, xsmic2n, xehfp)
        class(TableSet)      :: this
        integer             :: icomp
        real(4)             :: burn
        real(4)             :: xsmicd(ng,NISO), xsmica(ng,NISO), xsmicn(ng,NISO), xsmicf(ng,NISO), xsmick(ng,NISO), xsmics(ng,ng,NISO), xsmic2n(ng), xehfp(ng)

        call this%comps(icomp)%calculateReference(burn, xsmicd, xsmica, xsmicn, xsmicf, xsmick, xsmics, xsmic2n, xehfp)
        
    end subroutine
    
    subroutine calculateVariation(this, icomp, burn,   xdpmicn, xdfmicn, xdmmicn, xddmicn, &
                                                xdpmicf, xdfmicf, xdmmicf, xddmicf, &
                                                xdpmica, xdfmica, xdmmica, xddmica, &
                                                xdpmicd, xdfmicd, xdmmicd, xddmicd, &
                                                xdpmics, xdfmics, xdmmics, xddmics )
        class(TableSet)      :: this
        integer             :: icomp
        real(4)             :: burn
        real(4)             ::  xdpmicn(ng,NISO), xdfmicn(ng,NISO), xdmmicn(ng,3,NISO), xddmicn(ng,NISO), &
                                xdpmicf(ng,NISO), xdfmicf(ng,NISO), xdmmicf(ng,3,NISO), xddmicf(ng,NISO), &
                                xdpmica(ng,NISO), xdfmica(ng,NISO), xdmmica(ng,3,NISO), xddmica(ng,NISO), &
                                xdpmicd(ng,NISO), xdfmicd(ng,NISO), xdmmicd(ng,3,NISO), xddmicd(ng,NISO), &
                                xdpmics(ng,ng,NISO), xdfmics(ng,ng,NISO), xdmmics(ng,ng,3,NISO), xddmics(ng,ng,NISO)
        
        call this%comps(icomp)%calculateVariation(burn, xdpmicn, xdfmicn, xdmmicn, xddmicn, &
                                                xdpmicf, xdfmicf, xdmmicf, xddmicf, &
                                                xdpmica, xdfmica, xdmmica, xddmica, &
                                                xdpmicd, xdfmicd, xdmmicd, xddmicd, &
                                                xdpmics, xdfmics, xdmmics, xddmics )
        
    end subroutine    
    
    

end module
