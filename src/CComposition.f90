module CComposition
    use CIsotope
    
    implicit none
    private
    
    
    integer, public, parameter   :: LEN_COMPNAME = 13

    type, public    :: Composition
        character*(LEN_COMPNAME)    :: name
        type(Isotope)               :: iso(NISO)
        integer                     :: npoly(4)
        real(4)                     :: refvar(4)
        real(4)                     :: refpress
        real(4)                     :: b10ap
        
        integer                     :: npdet(2)
        real(4)                     :: xsend(NUM_BURN),xsn2n(NUM_BURN), chi(NUM_BURN,NUM_GRP), dpdet(NUM_BURN), dxsdet(NUM_BUVAR,NUM_GRP,NUM_BURN), df(NUM_BURN,NUM_GRP,3), ddf(NUM_BURN,NUM_GRP,3,3)
        integer                     :: nvar, nvar2
        real(4)                     :: xsbu(NUM_BURN), dxsbu(NUM_BUVAR)        
        integer                     :: nvarcr, nvar2cr
        real(4)                     :: xsbucr(NUM_BURN), dxsbucr(NUM_BUVAR)        
    
    contains
        procedure calculateReference
        procedure calculateVariation
    end type
    
contains

    subroutine calculateReference(this, burn, xsmicd, xsmica, xsmicn, xsmicf, xsmick, xsmics, xsmic2n, xehfp)
        class(Composition)  :: this
        real(XS_PREC)             :: burn
        real(XS_PREC)             :: xsmicd(NUM_GRP,NISO), xsmica(NUM_GRP,NISO), xsmicn(NUM_GRP,NISO), xsmicf(NUM_GRP,NISO), xsmick(NUM_GRP,NISO), xsmics(NUM_GRP,NUM_GRP,NISO), xsmic2n(NUM_GRP), xehfp
        integer             :: klo, i
        real(XS_PREC)             :: af(3)
        
        call quad1(this%nvar, this%xsbu(:), burn, klo, af)

        xsmic2n(1) = af(1) * this%xsn2n(klo) + af(2) * this%xsn2n(klo+1) + af(3) * this%xsn2n(klo+2)
        xehfp = af(1) * this%xsend(klo) + af(2) * this%xsend(klo+1) + af(3) * this%xsend(klo+2)
        
        do i = 1, NDEP
            call this%iso(i)%calculateReference(klo, af, xsmicd(:,i), xsmica(:,i), xsmicn(:,i), xsmicf(:,i), xsmick(:,i), xsmics(:,:,i))
        enddo
        
        call quad1(this%nvarcr, this%xsbucr(:), burn, klo, af)
        do i = ID_DEL1, ID_DEL3
            call this%iso(i)%calculateReference(klo, af, xsmicd(:,i), xsmica(:,i), xsmicn(:,i), xsmicf(:,i), xsmick(:,i), xsmics(:,:,i))
        enddo
        
    end subroutine
    
    subroutine calculateVariation(this, burn,   xdpmicn, xdfmicn, xdmmicn, xddmicn, &
                                                xdpmicf, xdfmicf, xdmmicf, xddmicf, &
                                                xdpmica, xdfmica, xdmmica, xddmica, &
                                                xdpmicd, xdfmicd, xdmmicd, xddmicd, &
                                                xdpmics, xdfmics, xdmmics, xddmics )
        class(Composition)  :: this
        real(XS_PREC)             :: burn
        real(XS_PREC)             ::  xdpmicn(NUM_GRP,NISO), xdfmicn(NUM_GRP,NISO), xdmmicn(NUM_GRP,3,NISO), xddmicn(NUM_GRP,NISO), &
                                xdpmicf(NUM_GRP,NISO), xdfmicf(NUM_GRP,NISO), xdmmicf(NUM_GRP,3,NISO), xddmicf(NUM_GRP,NISO), &
                                xdpmica(NUM_GRP,NISO), xdfmica(NUM_GRP,NISO), xdmmica(NUM_GRP,3,NISO), xddmica(NUM_GRP,NISO), &
                                xdpmicd(NUM_GRP,NISO), xdfmicd(NUM_GRP,NISO), xdmmicd(NUM_GRP,3,NISO), xddmicd(NUM_GRP,NISO), &
                                xdpmics(NUM_GRP,NUM_GRP,NISO), xdfmics(NUM_GRP,NUM_GRP,NISO), xdmmics(NUM_GRP,NUM_GRP,3,NISO), xddmics(NUM_GRP,NUM_GRP,NISO)
        integer             :: klo, i
        real(XS_PREC)             :: af(3)
        
        call quad1(this%nvar2, this%dxsbu(:), burn, klo, af)

        do i = 1, NDEP
            call this%iso(i)%calculateVariation(klo, af, xdpmicn(:,i), xdfmicn(:,i), xdmmicn(:,:,i), xddmicn(:,i), &
                                                         xdpmicf(:,i), xdfmicf(:,i), xdmmicf(:,:,i), xddmicf(:,i), &
                                                         xdpmica(:,i), xdfmica(:,i), xdmmica(:,:,i), xddmica(:,i), &
                                                         xdpmicd(:,i), xdfmicd(:,i), xdmmicd(:,:,i), xddmicd(:,i), &
                                                         xdpmics(:,:,i), xdfmics(:,:,i), xdmmics(:,:,:,i), xddmics(:,:,i))
        enddo
        
        call quad1(this%nvar2cr, this%dxsbucr(:), burn, klo, af)
        do i = ID_DEL1, ID_DEL3
            call this%iso(i)%calculateVariation(klo, af, xdpmicn(:,i), xdfmicn(:,i), xdmmicn(:,:,i), xddmicn(:,i), &
                                                         xdpmicf(:,i), xdfmicf(:,i), xdmmicf(:,:,i), xddmicf(:,i), &
                                                         xdpmica(:,i), xdfmica(:,i), xdmmica(:,:,i), xddmica(:,i), &
                                                         xdpmicd(:,i), xdfmicd(:,i), xdmmicd(:,:,i), xddmicd(:,i), &
                                                         xdpmics(:,:,i), xdfmics(:,:,i), xdmmics(:,:,:,i), xddmics(:,:,i))
        enddo
        
    end subroutine    

end module
