module CTableSet
    
    implicit none
    private
    
    
    integer, parameter   :: ns = 60
    integer, parameter   :: nv = 12
    integer, parameter   :: LEN_COMPNAME = 12
    integer, parameter   :: LEN_ISOTOPE = 4
    integer, parameter   :: NISO = 40
    character(len=4),parameter, dimension(NISO) :: NNUID =   (/'U234', 'U235', 'U236', 'NP37', 'U238', &
                                                               'PU48', 'NP39', 'PU49', 'PU40', 'PU41', &
                                                               'PU42', 'AM43', 'RESI', 'POIS', 'PM47', &
                                                               'PS48', 'PM48', 'PM49', 'SM29', 'I135', &
                                                               'XE45', 'FP.1', 'B-10', 'H2O ', 'STRM', &
                                                               'AM41', 'AM42', 'CM42', 'CM44', 'TH32', &
                                                               'PA33', 'U233', 'MAC ', 'DEL1', 'DEL2', &
                                                               'DEL3', 'TMOD', 'DETE', '   V', 'XSE '/)    
                                                               
    type, public    :: Isotope
        character*(LEN_ISOTOPE)     :: name
        integer                     :: nax, nvar,iwab,nskip,nbltab(5)
    end type
                                                               
    
    type, public    :: Composition
        character*(LEN_COMPNAME)    :: name
        integer                     :: npoly(4)
        real(4)                     :: refvar(4)
        real(4)                     :: refpress
        real(4)                     :: b10ap
        
    end type
    
    type, public   :: TableSet
        integer             :: nrefl
        integer             :: ncomp
        integer, pointer    :: icomps(:)
        character(len=LEN_COMPNAME), pointer    :: compnames(:)
    contains
        procedure readFile
        procedure readReflComp
        procedure readFuelComp
    end type

contains

    subroutine init(this, ncomp, icomps, compnames)
        class(TableSet) :: this
        integer         :: ncomp
        integer         :: icomps(ncomp)
        character*(LEN_COMPNAME)   :: compnames(ncomp)
        
        integer         :: i
        
        this.ncomp = ncomp
        
        allocate(this.icomps(ncomp))
        this.icomps = icomps
        
        allocate(this.compnames(ncomp))
        
        do i = 1, ncomp
            this.compnames(i) = compnames(i)
        enddo
        
    end subroutine

    subroutine readFile(this, filename)
        class(TableSet) :: this
        character*(*)   :: filename
    
        
    end subroutine

    subroutine readReflComp(this, icomp)
        class(TableSet) :: this
        integer         :: icomp
    
    end subroutine
    
    subroutine readFuelComp(this, ifile, comp, header)
        class(TableSet) :: this
        integer         :: ifile
        type(Composition)   :: comp
        character*(*)   :: header
        
        READ (header, '(5X,A12,39X,4I4)') comp%NMXSET, comp%npoly(1:4)
         
        READ (ifile,'(6F12.6)')  comp%refvar(1:4), comp%refpress, comp%b10ap
        if (comp%b10ap == 0) b10ap = 19.8

        !b10wp1  = b10abnd*b10aw / (b10abnd*b10aw + (100.-b10abnd)*b11aw)
        !ppmcorr = b10wp1*100.0 / b10wp
        !refppm(ixset) = ppmcorr * refppm(ixset)
    
    end subroutine
    
    
    
    subroutine readFuelIsotope(this, ifile, comp, header)
        class(TableSet)     :: this
        integer             :: ifile
        type(Composition)   :: comp
        character*(*)       :: header
    
        integer             :: iiso
        logical             :: n2n
        character*(4)       :: nd
        character*(44)      :: titl
        integer             :: nax, nvar, iwab, nbltab(5), nskip
        
        iiso = this%indexOfIsotope(header(1:4))
        
        IF(header(5:7) == 'N2N')) N2N = .TRUE.
        
        read(header,1) nd,titl,nax,nvar,iwab,nbltab(1),nskip, (nbltab(i),i=2,5)
        
        call skip(ifile, nskip)
        
        if(nvar.lt.0) nvar=-nvar
        
        read(ifile,62) ppm,tfuel,tmod,dm,pressxs,void
        
        lin1 = nvar/6
        ires = mod(nvar,6)
        if(ires.gt.0) lin1=lin1+1
        
        do i=1,lin1
            ib=(i-1)*6+1
            ie=ib+5
            if(ie.gt.nvar) ie=nvar
            read(ifile,2) (xsbu(j,ind,ixset),j=ib,ie)
        enddo
        
        if(nd(1:3) == 'XSE')) then
            do i1=1,lin1
                ib=(i1-1)*6+1
                ie=ib+5
                if(ie.gt.nvar) ie=nvar
                read(inp,2) (xsend(j,ixset),j=ib,ie)
            enddo
        endif

         IF(NBLTAB(1).GT.0) READ(INP,62) (CAPPA(I,IND,IXSET),I=1,NG)

         IF(eqstr(ND(1:4), 'U238') .AND. N2N) THEN
            DO I1=1,LIN1
               IB=(I1-1)*6+1
               IE=IB+5
               IF(IE.GT.NVAR) IE=NVAR
               READ(INP,62) (XSN2N(J,IXSET),J=IB,IE)
            ENDDO         
         ENDIF

         IF(eqstr(ND(1:3), 'MAC')) THEN
            DO IG=1,NG
               DO I1=1,LIN1
                  IB=(I1-1)*6+1
                  IE=IB+5
                  IF(IE.GT.NVAR) IE=NVAR
                  READ(INP,62) (CHIT(J,IG,IXSET),J=IB,IE)
               ENDDO
            ENDDO
         ENDIF
!        write values on xscomp
!        ixset : no. of composition
!        ind   : sequential no. of isotope

         ndxs(1,ind,ixset) = nvar
         ndxs(2,ind,ixset) = nvar2

         ! the order of xsec types : 
         ! nu-fission(1), fission(2), capture(3), transport(4,skipped)
         do ixs=1,nxs

            if(nbltab(ixs).eq.0) then
               nsig(ixs,ind,ixset) = 0
               cycle
            endif
            
            nsig(ixs,ind,ixset) = ixs + (ind-1) * 4 + (ixset-1) * 4 * nnucl

            do ig=1,ng
               do i1=1,lin1
                  ib=(i1-1)*6+1
                  ie=ib+5
                  if(ie.gt.nvar) ie=nvar
                  read(inp,62) (xsig(j,ig,ixs, ind, ixset),j=ib,ie)
               enddo
            enddo
            
            ! 1111 --> from right, 1st;ppm, 2nd;tf, 3rd;tm, 4th;dm
            if(nbltab(ixs).eq.16) cycle
         
            mok=nbltab(ixs)            
            do ivar=1,ndrv
               if(mod(mok,2).eq.1) then
                  do ig=1,ng
                  do ip = 0, npoly(ivar,ixset)
                  do i1=1,lin2
                     ib=(i1-1)*6+1
                     ie=ib+5
                     if(ie.gt.nvar2) ie=nvar2
                     read(inp,62) (dxsig(j,ig,ip,ivar,ixs,ind,ixset),j=ib,ie)
                  enddo
                  enddo
                  enddo
                  if(npoly(ivar,ixset) .eq. 0) then
                     dxsig(:,:,1,ivar,ixs,ind,ixset) = dxsig(:,:,0,ivar,ixs,ind,ixset)
                     dxsig(:,:,0,ivar,ixs,ind,ixset) = 0.0
                  endif
               endif
               mok=mok*0.5
               if(mok.le.0) exit
            enddo

            dxsig(:,:,:,1,ixs, ind, ixset) = ppmcorr * dxsig(:,:,:,1,ixs, ind, ixset)

         enddo ! ixs

         ! scattering xs
         ixs=5
         if(nbltab(ixs).gt.0) then
            nsigs(ind,ixset) = ind+ (ixset-1) * nnucl

            do igs = 1, ng
            do ige = 1, ng
               if(igs.eq.ige) cycle ! skip self-scattering
               do i1=1,lin1
                  ib=(i1-1)*6+1
                  ie=ib+5
                   if(ie.gt.nvar) ie=nvar
                   read(inp,62) (xsigs(j,igs,ige,ind,ixset),j=ib,ie)
               enddo
            enddo
            enddo

            if(nbltab(ixs).ne.16) then
               mok=nbltab(ixs)

               ! variation type : ppm, tf, tm, dm
               do ivar=1,ndrv
                  ires = mod(mok,2)
                  if(ires.eq.1) then
                     do igs=1,ng
                     do ige=1,ng
                     do ip = 0, npoly(ivar, ixset)
                        if(igs.eq.ige) cycle
                        do i1=1,lin2
                           ib=(i1-1)*6+1
                           ie=ib+5
                           if(ie.gt.nvar2) ie=nvar2
                           read(inp,62) (dxsigs(j,igs,ige,ip,ivar,ind,ixset),j=ib,ie)
                        enddo
                     enddo
                     enddo
                     enddo
                  endif
                  if(npoly(ivar,ixset) .eq. 0) then
                     dxsigs(:,:,:,1,ivar,ind,ixset) = dxsigs(:,:,:,0,ivar,ind,ixset)
                     dxsigs(:,:,:,0,ivar,ind,ixset) = 0.0
                  endif
               
                  mok=mok*0.5
                  if(mok .le. 0) exit
               enddo

               dxsigs(:,:,:,:,1,ind,ixset) = ppmcorr * dxsigs(:,:,:,:,1,ind,ixset)

            endif
         endif  ! nbltab(i).gt.0
! yji - scattering xs
      
      endif  ! if(ncomp)
               
1       FORMAT(A4,A44,I1,I3,I1,1X,I2,I1,1X,I2,3I4)        
2       FORMAT(6E12.6)
    end subroutine
    
    
    function indexOfIsotope(this, nmiso) 
        class(TableSet)     :: this
        integer             :: iiso
        character*(4)       :: nmiso
        
        do iiso = 1, NISO
            if(nmiso .eq. NNUID(iiso)) exit;
        enddo
        
    
    end function

    subroutine skip(ifile,nline)
        do i=1,nline
            read(ifile,'(a)') 
        enddo
    end subroutine

end module
    