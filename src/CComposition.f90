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
        real(4)                     :: xsend(ns),xsn2n(ns), chi(ns,ng), dpdet(ns), dxsdet(nv,ng,ns), df(ns,ng,3), ddf(ns,ng,3,3)
        integer                     :: nvar, nvar2
        real(4)                     :: xsbu(ns), dxsbu(nv)        
        integer                     :: nvarcr, nvar2cr
        real(4)                     :: xsbucr(ns), dxsbucr(nv)        
    
    contains
        procedure calculateReference
        procedure calculateVariation
    end type
    
contains

    subroutine calculateReference(this, burn, xsmicd, xsmica, xsmicn, xsmicf, xsmick, xsmics, xsmic2n, xehfp)
        class(Composition)  :: this
        real(4)             :: burn
        real(4)             :: xsmicd(ng,NISO), xsmica(ng,NISO), xsmicn(ng,NISO), xsmicf(ng,NISO), xsmick(ng,NISO), xsmics(ng,ng,NISO), xsmic2n(ng), xehfp
        integer             :: klo, i
        real(4)             :: af(3)
        
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
        real(4)             :: burn
        real(4)             ::  xdpmicn(ng,NISO), xdfmicn(ng,NISO), xdmmicn(ng,3,NISO), xddmicn(ng,NISO), &
                                xdpmicf(ng,NISO), xdfmicf(ng,NISO), xdmmicf(ng,3,NISO), xddmicf(ng,NISO), &
                                xdpmica(ng,NISO), xdfmica(ng,NISO), xdmmica(ng,3,NISO), xddmica(ng,NISO), &
                                xdpmicd(ng,NISO), xdfmicd(ng,NISO), xdmmicd(ng,3,NISO), xddmicd(ng,NISO), &
                                xdpmics(ng,ng,NISO), xdfmics(ng,ng,NISO), xdmmics(ng,ng,3,NISO), xddmics(ng,ng,NISO)
        integer             :: klo, i
        real(4)             :: af(3)
        
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
