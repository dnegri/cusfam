module CIsotope
    implicit none
    private
    
    integer, public, parameter   :: NUM_BURN = 60
    integer, public, parameter   :: NUM_BUVAR = 12
    integer, public, parameter   :: LEN_ISOTOPE = 4
    integer, public, parameter   :: NISO = 40
    integer, public, parameter   :: NDEP = 25
    integer, public, parameter   :: NUM_GRP = 2
    integer, public, parameter   :: NUM_GRPSCT = (NUM_GRP-1)*NUM_GRP !skip self-scattering 
    integer, public, parameter   :: XSSIG_NTYPE = 4
    integer, public, parameter   :: XSSIG_NUF = 1
    integer, public, parameter   :: XSSIG_FIS = 2
    integer, public, parameter   :: XSSIG_CAP = 3
    integer, public, parameter   :: XSSIG_TRS = 4
    integer, public, parameter   :: VAR_NTYPE = 4
    integer, public, parameter   :: VAR_PPM = 1
    integer, public, parameter   :: VAR_TF = 2
    integer, public, parameter   :: VAR_TM = 3
    integer, public, parameter   :: VAR_DM = 4
    integer, public, parameter   :: VAR_NPOLY = 3
    integer, public, parameter   :: VAR_NTNP = XSSIG_NTYPE*(VAR_NPOLY+1)
    
    
    character(len=4),public, parameter, dimension(NISO) :: NNUID =   (/'U234', 'U235', 'U236', 'NP37', 'U238', &
                                                               'PU48', 'NP39', 'PU49', 'PU40', 'PU41', &
                                                               'PU42', 'AM43', 'RESI', 'POIS', 'PM47', &
                                                               'PS48', 'PM48', 'PM49', 'SM29', 'I135', &
                                                               'XE45', 'FP.1', 'B-10', 'H2O ', 'STRM', &
                                                               'AM41', 'AM42', 'CM42', 'CM44', 'TH32', &
                                                               'PA33', 'U233', 'MAC ', 'DEL1', 'DEL2', &
                                                               'DEL3', 'TMOD', 'DETE', '   V', 'XSE '/)
    integer, public, parameter :: ID_U234 = 1
    integer, public, parameter :: ID_U235 = 2
    integer, public, parameter :: ID_U236 = 3
    integer, public, parameter :: ID_NP37 = 4
    integer, public, parameter :: ID_U238 = 5
    integer, public, parameter :: ID_PU48 = 6
    integer, public, parameter :: ID_NP39 = 7
    integer, public, parameter :: ID_PU49 = 8
    integer, public, parameter :: ID_PU40 = 9
    integer, public, parameter :: ID_PU41 = 10
    integer, public, parameter :: ID_PU42 = 11
    integer, public, parameter :: ID_AM43 = 12
    integer, public, parameter :: ID_RESI = 13
    integer, public, parameter :: ID_POIS = 14
    integer, public, parameter :: ID_PM47 = 15
    integer, public, parameter :: ID_PS48 = 16
    integer, public, parameter :: ID_PM48 = 17
    integer, public, parameter :: ID_PM49 = 18
    integer, public, parameter :: ID_SM29 = 19
    integer, public, parameter :: ID_I135 = 20
    integer, public, parameter :: ID_XE45 = 21
    integer, public, parameter :: ID_FP1  = 22
    integer, public, parameter :: ID_B10  = 23
    integer, public, parameter :: ID_H2O  = 24
    integer, public, parameter :: ID_STRM = 25
    integer, public, parameter :: ID_AM41 = 26
    integer, public, parameter :: ID_AM42 = 27
    integer, public, parameter :: ID_CM42 = 28
    integer, public, parameter :: ID_CM44 = 29
    integer, public, parameter :: ID_TH32 = 30
    integer, public, parameter :: ID_PA33 = 31
    integer, public, parameter :: ID_U233 = 32
    integer, public, parameter :: ID_MAC  = 33
    integer, public, parameter :: ID_DEL1 = 34
    integer, public, parameter :: ID_DEL2 = 35
    integer, public, parameter :: ID_DEL3 = 36
    integer, public, parameter :: ID_TMOD = 37
    integer, public, parameter :: ID_DETE = 38
    integer, public, parameter :: ID_V    = 39
    integer, public, parameter :: ID_XSE  = 40
    
    type, public    :: Isotope
        integer                     :: id
        integer                     :: nax, iwab,nskip,nbltab(5)
        real(4)                     :: cappa(NUM_GRP)
        real(4)                     :: xssig(NUM_BURN, NUM_GRP, XSSIG_NTYPE), dxssig(NUM_BUVAR, NUM_GRP, VAR_NTNP, XSSIG_NTYPE)
        real(4)                     :: xsigs(NUM_BURN, NUM_GRPSCT), dxsigs(NUM_BUVAR, NUM_GRPSCT, VAR_NTNP)
    contains
        procedure calculateReference
        procedure calculateVariation
        procedure, private :: calculateVariation1
    end type
                                                               

contains

    subroutine calculateReference(this, klo, af, xsmicd, xsmica, xsmicn, xsmicf, xsmick, xsmics)
        class(Isotope)     :: this
        integer             :: klo
        real(XS_PREC)       :: af(3)        
        real(XS_PREC)       ::  xsmicd(NUM_GRP), xsmica(NUM_GRP), xsmicn(NUM_GRP), xsmicf(NUM_GRP), xsmick(NUM_GRP), xsmics(NUM_GRP,NUM_GRP)
        integer             :: ixs, ig, igs, ige, igse
        
        ixs = 1
        do ig=1,NUM_GRP
            xsmicn(ig) = af(1) * this%xssig(klo,ig,ixs) + af(2) * this%xssig(klo+1,ig,ixs) + af(3) * this%xssig(klo+2,ig,ixs)
        enddo
        ixs = 2
        do ig=1,NUM_GRP
            xsmicf(ig) = af(1) * this%xssig(klo,ig,ixs) + af(2) * this%xssig(klo+1,ig,ixs) + af(3) * this%xssig(klo+2,ig,ixs)
            xsmick(ig) = xsmicf(ig)*this%cappa(ig)
        enddo
        ixs = 3
        do ig=1,NUM_GRP
            xsmica(ig) = af(1) * this%xssig(klo,ig,ixs) + af(2) * this%xssig(klo+1,ig,ixs) + af(3) * this%xssig(klo+2,ig,ixs)
            xsmica(ig) = xsmica(ig) + xsmicf(ig)
        enddo
        ixs = 4
        do ig=1,NUM_GRP
            xsmicd(ig) = af(1) * this%xssig(klo,ig,ixs) + af(2) * this%xssig(klo+1,ig,ixs) + af(3) * this%xssig(klo+2,ig,ixs)
        enddo
        
        igse = 0
        do igs=1,NUM_GRP
        do ige=1,NUM_GRP
            if(igs.eq.ige) cycle
            igse = igse + 1
            xsmics(igs,ige) = af(1) * this%xsigs(klo,igse) + af(2) * this%xsigs(klo+1,igse) + af(3) * this%xsigs(klo+2,igse)
        enddo
        enddo
        
    end subroutine
    
    
    subroutine calculateVariation(this, klo, af,xdpmicn, xdfmicn, xdmmicn, xddmicn, &
                                                xdpmicf, xdfmicf, xdmmicf, xddmicf, &
                                                xdpmica, xdfmica, xdmmica, xddmica, &
                                                xdpmicd, xdfmicd, xdmmicd, xddmicd, &
                                                xdpmics, xdfmics, xdmmics, xddmics)
        class(Isotope)      :: this
        integer             :: klo
        real(XS_PREC)             :: af(3)
        real(XS_PREC)             ::  xdpmicn(NUM_GRP), xdfmicn(NUM_GRP), xdmmicn(NUM_GRP,3), xddmicn(NUM_GRP), &
                                xdpmicf(NUM_GRP), xdfmicf(NUM_GRP), xdmmicf(NUM_GRP,3), xddmicf(NUM_GRP), &
                                xdpmica(NUM_GRP), xdfmica(NUM_GRP), xdmmica(NUM_GRP,3), xddmica(NUM_GRP), &
                                xdpmicd(NUM_GRP), xdfmicd(NUM_GRP), xdmmicd(NUM_GRP,3), xddmicd(NUM_GRP), &
                                xdpmics(NUM_GRP,NUM_GRP), xdfmics(NUM_GRP,NUM_GRP), xdmmics(NUM_GRP,NUM_GRP,3), xddmics(NUM_GRP,NUM_GRP)
        integer             :: ig

        call this%calculateVariation1(klo, af, 2, xdpmicn(:), xdpmicf(:), xdpmica(:), xdpmicd(:), xdpmics(:,:))
        call this%calculateVariation1(klo, af, 4, xdfmicn(:), xdfmicf(:), xdfmica(:), xdfmicd(:), xdfmics(:,:))
        call this%calculateVariation1(klo, af, 6, xdmmicn(:,1), xdmmicf(:,1), xdmmica(:,1), xdmmicd(:,1), xdmmics(:,:,1))
        call this%calculateVariation1(klo, af, 7, xdmmicn(:,2), xdmmicf(:,2), xdmmica(:,2), xdmmicd(:,2), xdmmics(:,:,2))
        call this%calculateVariation1(klo, af, 8, xdmmicn(:,3), xdmmicf(:,3), xdmmica(:,3), xdmmicd(:,3), xdmmics(:,:,3))
        call this%calculateVariation1(klo, af,10, xddmicn(:), xddmicf(:), xddmica(:), xddmicd(:), xddmics(:,:))
        
        
    end subroutine    
    
    subroutine calculateVariation1(this, klo, af, idxdrv, xdmicn, xdmicf, xdmica, xdmicd, xdmics)
    
        class(Isotope)     :: this
        integer             :: klo, idxdrv
        real(XS_PREC)             :: af(3)
        real(XS_PREC)             ::  xdmicn(NUM_GRP), xdmicf(NUM_GRP), xdmica(NUM_GRP), xdmicd(NUM_GRP), xdmics(NUM_GRP,NUM_GRP)
        integer             :: ixs,ig, igs, ige, igse
        
        ixs = XSSIG_NUF
        do ig=1,NUM_GRP
            xdmicn(ig) = af(1) * this%dxssig(klo,ig,idxdrv,ixs) + af(2) * this%dxssig(klo+1,ig,idxdrv,ixs) + af(3) * this%dxssig(klo+2,ig,idxdrv,ixs)
        enddo
        
        ixs = XSSIG_FIS
        do ig=1,NUM_GRP
            xdmicf(ig) = af(1) * this%dxssig(klo,ig,idxdrv,ixs) + af(2) * this%dxssig(klo+1,ig,idxdrv,ixs) + af(3) * this%dxssig(klo+2,ig,idxdrv,ixs)
        enddo
        
        ixs = XSSIG_CAP
        do ig=1,NUM_GRP
            xdmica(ig) = af(1) * this%dxssig(klo,ig,idxdrv,ixs) + af(2) * this%dxssig(klo+1,ig,idxdrv,ixs) + af(3) * this%dxssig(klo+2,ig,idxdrv,ixs)
            xdmica(ig) = xdmica(ig) + xdmicf(ig)
        enddo
        
        igse = 0
        do igs=1,NUM_GRP
        do ige=1,NUM_GRP
            if(igs.eq.ige) cycle
            igse = igse + 1
            xdmics(igs,ige) = af(1) * this%dxsigs(klo,igse,idxdrv) + af(2) * this%dxsigs(klo+1,igse,idxdrv) + af(3) * this%dxsigs(klo+2,igse,idxdrv)
        enddo
        enddo
        
    end subroutine    
    


end module
