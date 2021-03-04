module CReflector
    use CIsotope
    implicit none
    private
    
    integer, public, parameter :: REFL_BOTTOM   = -1
    integer, public, parameter :: REFL_TOP      = -2
    integer, public, parameter :: REFL_CORNER   = -3
    integer, public, parameter :: REFL_EDGE     = -4
    integer, public, parameter :: REFL_NTYPE    =  4
                                                               
                                                               
    type, public    :: Reflector
        real(4)                     :: rasigb(4,ng,3), rasigsb(ng,ng,3)
        real(4)                     :: rasigt(4,ng,3), rasigst(ng,ng,3)
        real(4)                     :: rrsig(4,ng,4), rrsigs(2,ng,ng,2)
        real(4)                     :: rfrppm(REFL_EDGE:REFL_BOTTOM), rfrtf(REFL_EDGE:REFL_BOTTOM), rfrtm(REFL_EDGE:REFL_BOTTOM), &
                                        rfrdm(REFL_EDGE:REFL_BOTTOM) ,rfrprs(REFL_EDGE:REFL_BOTTOM), rfratm(REFL_EDGE:REFL_BOTTOM), b10ap(REFL_EDGE:REFL_BOTTOM)
    contains
        procedure calculate
        procedure calculateBottom
        procedure calculateTop
        procedure calculateCorner
        procedure calculateEdge
    end type
       

contains

    subroutine calculate(this, reflType, xsmica, xsmicd, xsmics, xdpmica, xdmmica, xddmica, &
                                                           xdpmicd, xdmmicd, xddmicd, &
                                                           xdpmics, xdmmics, xddmics )
        class(Reflector)    :: this
        integer             :: reflType
        real(4)             ::  xsmicd(ng,NISO), xsmica(ng,NISO), xsmics(ng,ng,NISO)
        real(4)             ::  xdpmica(ng,NISO), xdmmica(ng,3,NISO), xddmica(ng,NISO), &
                                xdpmicd(ng,NISO), xdmmicd(ng,3,NISO), xddmicd(ng,NISO), &
                                xdpmics(ng,ng,NISO), xdmmics(ng,ng,3,NISO), xddmics(ng,ng,NISO)
        

        select case(reflType)
        case(REFL_BOTTOM)
            call this%calculateBottom(xsmica, xsmicd, xsmics)
        case(REFL_TOP)
            call this%calculateTop(xsmica, xsmicd, xsmics)
        case(REFL_CORNER)
            call this%calculateCorner(  xsmica, xsmicd, xsmics, &
                                        xdpmica, xdmmica, xddmica, &
                                        xdpmicd, xdmmicd, xddmicd, &
                                        xdpmics, xdmmics, xddmics )
        case(REFL_EDGE)
            call this%calculateEdge(    xsmica, xsmicd, xsmics,   &
                                        xdpmica, xdmmica, xddmica, &
                                        xdpmicd, xdmmicd, xddmicd, &
                                        xdpmics, xdmmics, xddmics )
        
        end select
    end subroutine
    
    subroutine calculateBottom(this, xsmica, xsmicd, xsmics)
        class(Reflector)    :: this
        real(4)             ::  xsmicd(ng,NISO), xsmica(ng,NISO), xsmics(ng,ng,NISO)
        integer             :: ig, igs, ige
        do ig=1,ng
            xsmica(ig, ID_B10)=this%rasigb(1,ig,1)
            xsmicd(ig, ID_B10)=this%rasigb(2,ig,1)
            xsmica(ig, ID_H2O)=this%rasigb(1,ig,2)
            xsmicd(ig, ID_H2O)=this%rasigb(2,ig,2)
            xsmica(ig,ID_STRM)=this%rasigb(1,ig,3)
            xsmicd(ig,ID_STRM)=this%rasigb(2,ig,3)
        enddo
        
        do igs=1,ng
        do ige=1,ng
            if(igs == ige) cycle
            xsmics(igs,ige, ID_B10)=this%rasigsb(igs,ige,1)
            xsmics(igs,ige, ID_H2O)=this%rasigsb(igs,ige,2)
            xsmics(igs,ige,ID_STRM)=this%rasigsb(igs,ige,3)
        enddo
        enddo
        
    end subroutine
    
    subroutine calculateTop(this, xsmica, xsmicd, xsmics)
        class(Reflector)    :: this
        real(4)             ::  xsmicd(ng,NISO), xsmica(ng,NISO), xsmics(ng,ng,NISO)
        integer             :: ig, igs, ige        
        do ig=1,ng
            xsmica(ig, ID_B10)=this%rasigt(1,ig,1)
            xsmicd(ig, ID_B10)=this%rasigt(2,ig,1)
            xsmica(ig, ID_H2O)=this%rasigt(1,ig,2)
            xsmicd(ig, ID_H2O)=this%rasigt(2,ig,2)
            xsmica(ig,ID_STRM)=this%rasigt(1,ig,3)
            xsmicd(ig,ID_STRM)=this%rasigt(2,ig,3)
        enddo
        
        do igs=1,ng
        do ige=1,ng
            if(igs == ige) cycle
            xsmics(igs,ige, ID_B10)=this%rasigst(igs,ige,1)
            xsmics(igs,ige, ID_H2O)=this%rasigst(igs,ige,2)
            xsmics(igs,ige,ID_STRM)=this%rasigst(igs,ige,3)
        enddo
        enddo
        
    end subroutine    
    
    subroutine calculateEdge(this, xsmica, xsmicd, xsmics, xdpmica, xdmmica, xddmica, &
                                                           xdpmicd, xdmmicd, xddmicd, &
                                                           xdpmics, xdmmics, xddmics )
        class(Reflector)    :: this
        real(4)             ::  xsmicd(ng,NISO), xsmica(ng,NISO), xsmics(ng,ng,NISO)
        real(4)             ::  xdpmica(ng,NISO), xdmmica(ng,3,NISO), xddmica(ng,NISO), &
                                xdpmicd(ng,NISO), xdmmicd(ng,3,NISO), xddmicd(ng,NISO), &
                                xdpmics(ng,ng,NISO), xdmmics(ng,ng,3,NISO), xddmics(ng,ng,NISO)
        integer             :: ig, igs, ige        
        do ig=1,ng
            xsmica(ig,ID_STRM)=this%rrsig(1,ig,3)
            xsmicd(ig,ID_STRM)=this%rrsig(1,ig,4)
        enddo
            
        do igs=1,ng
        do ige=1,ng
            if(igs.eq.ige) cycle
            xsmics(igs,ige,ID_STRM)=this%rrsigs(1,igs,ige,2)
        enddo
        enddo
        
        do ig=1,ng
            xdpmica(ig,ID_STRM)=this%rrsig(2,ig,3)
            xdpmicd(ig,ID_STRM)=this%rrsig(2,ig,4)
            xdmmica(ig,1,ID_STRM)=this%rrsig(3,ig,3)
            xdmmicd(ig,1,ID_STRM)=this%rrsig(3,ig,4)
            xddmica(ig,ID_STRM)=this%rrsig(4,ig,3)
            xddmicd(ig,ID_STRM)=this%rrsig(4,ig,4)
        enddo
        
        do igs=1,ng
        do ige=1,ng
            if(igs.eq.ige) cycle
            xdpmics(igs,ige,ID_STRM)=this%rrsigs(2,igs,ige,2)
        enddo
        enddo
        
    end subroutine       
    
    subroutine calculateCorner(this, xsmica, xsmicd, xsmics, xdpmica, xdmmica, xddmica, &
                                                           xdpmicd, xdmmicd, xddmicd, &
                                                           xdpmics, xdmmics, xddmics )
        class(Reflector)    :: this
        real(4)             ::  xsmicd(ng,NISO), xsmica(ng,NISO), xsmics(ng,ng,NISO)
        real(4)             ::  xdpmica(ng,NISO), xdmmica(ng,3,NISO), xddmica(ng,NISO), &
                                xdpmicd(ng,NISO), xdmmicd(ng,3,NISO), xddmicd(ng,NISO), &
                                xdpmics(ng,ng,NISO), xdmmics(ng,ng,3,NISO), xddmics(ng,ng,NISO)
        integer             :: ig, igs, ige        
        do ig=1,ng
            xsmica(ig,ID_STRM)=this%rrsig(1,ig,1)
            xsmicd(ig,ID_STRM)=this%rrsig(1,ig,2)
        enddo
            
        do igs=1,ng
        do ige=1,ng
            if(igs == ige) cycle
            xsmics(igs,ige,ID_STRM)=this%rrsigs(1,igs,ige,1)
        enddo
        enddo
        
        do ig=1,ng
            xdpmica(ig,ID_STRM)=this%rrsig(2,ig,1)
            xdpmicd(ig,ID_STRM)=this%rrsig(2,ig,2)
            xdmmica(ig,1,ID_STRM)=this%rrsig(3,ig,1)
            xdmmicd(ig,1,ID_STRM)=this%rrsig(3,ig,2)
            xddmica(ig,ID_STRM)=this%rrsig(4,ig,1)
            xddmicd(ig,ID_STRM)=this%rrsig(4,ig,2)
        enddo
        
        do igs=1,ng
        do ige=1,ng
            if(igs.eq.ige) cycle
            xdpmics(igs,ige,ID_STRM)=this%rrsigs(2,igs,ige,1)
        enddo
        enddo
        
        
    end subroutine      

end module
