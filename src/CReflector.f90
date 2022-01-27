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
        real(4)                     :: rasigb(4,NUM_GRP,3), rasigsb(NUM_GRP,NUM_GRP,3)
        real(4)                     :: rasigt(4,NUM_GRP,3), rasigst(NUM_GRP,NUM_GRP,3)
        real(4)                     :: rrsig(4,NUM_GRP,4), rrsigs(2,NUM_GRP,NUM_GRP,2)
        real(4)                     :: rfrppm(REFL_EDGE:REFL_BOTTOM), rfrtf(REFL_EDGE:REFL_BOTTOM), rfrtm(REFL_EDGE:REFL_BOTTOM), &
                                        rfrdm(REFL_EDGE:REFL_BOTTOM) ,rfrprs(REFL_EDGE:REFL_BOTTOM), rfratm(REFL_EDGE:REFL_BOTTOM), b10ap(REFL_EDGE:REFL_BOTTOM)
    contains
        procedure init
        procedure calculate
        procedure calculateBottom
        procedure calculateTop
        procedure calculateCorner
        procedure calculateEdge
    end type
       

contains
    subroutine init(this)
        class(Reflector)    :: this
        
        this%rasigb = 0.0
        this%rasigsb = 0.0
        this%rasigt = 0.0
        this%rasigst = 0.0
        this%rrsig = 0.0
        this%rrsigs = 0.0
        this%rfrppm = 0.0
        this%rfrtf = 0.0
        this%rfrtm = 0.0
        this%rfrdm = 0.0
        this%rfrprs = 0.0
        this%rfratm = 0.0
    end subroutine

    subroutine calculate(this, reflType, xsmica, xsmicd, xsmics, xdpmica, xdmmica, xddmica, &
                                                           xdpmicd, xdmmicd, xddmicd, &
                                                           xdpmics, xdmmics, xddmics )
        class(Reflector)    :: this
        integer             :: reflType
        real(XS_PREC)             ::  xsmicd(NUM_GRP,NISO), xsmica(NUM_GRP,NISO), xsmics(NUM_GRP,NUM_GRP,NISO)
        real(XS_PREC)             ::  xdpmica(NUM_GRP,NISO), xdmmica(NUM_GRP,3,NISO), xddmica(NUM_GRP,NISO), &
                                xdpmicd(NUM_GRP,NISO), xdmmicd(NUM_GRP,3,NISO), xddmicd(NUM_GRP,NISO), &
                                xdpmics(NUM_GRP,NUM_GRP,NISO), xdmmics(NUM_GRP,NUM_GRP,3,NISO), xddmics(NUM_GRP,NUM_GRP,NISO)
        

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
        real(XS_PREC)             ::  xsmicd(NUM_GRP,NISO), xsmica(NUM_GRP,NISO), xsmics(NUM_GRP,NUM_GRP,NISO)
        integer             :: ig, igs, ige
        do ig=1,NUM_GRP
            xsmica(ig, ID_B10)=this%rasigb(1,ig,1)
            xsmicd(ig, ID_B10)=this%rasigb(2,ig,1)
            xsmica(ig, ID_H2O)=this%rasigb(1,ig,2)
            xsmicd(ig, ID_H2O)=this%rasigb(2,ig,2)
            xsmica(ig,ID_STRM)=this%rasigb(1,ig,3)
            xsmicd(ig,ID_STRM)=this%rasigb(2,ig,3)
        enddo
        
        do igs=1,NUM_GRP
        do ige=1,NUM_GRP
            if(igs == ige) cycle
            xsmics(igs,ige, ID_B10)=this%rasigsb(igs,ige,1)
            xsmics(igs,ige, ID_H2O)=this%rasigsb(igs,ige,2)
            xsmics(igs,ige,ID_STRM)=this%rasigsb(igs,ige,3)
        enddo
        enddo
        
    end subroutine
    
    subroutine calculateTop(this, xsmica, xsmicd, xsmics)
        class(Reflector)    :: this
        real(XS_PREC)             ::  xsmicd(NUM_GRP,NISO), xsmica(NUM_GRP,NISO), xsmics(NUM_GRP,NUM_GRP,NISO)
        integer             :: ig, igs, ige        
        do ig=1,NUM_GRP
            xsmica(ig, ID_B10)=this%rasigt(1,ig,1)
            xsmicd(ig, ID_B10)=this%rasigt(2,ig,1)
            xsmica(ig, ID_H2O)=this%rasigt(1,ig,2)
            xsmicd(ig, ID_H2O)=this%rasigt(2,ig,2)
            xsmica(ig,ID_STRM)=this%rasigt(1,ig,3)
            xsmicd(ig,ID_STRM)=this%rasigt(2,ig,3)
        enddo
        
        do igs=1,NUM_GRP
        do ige=1,NUM_GRP
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
        real(XS_PREC)             ::  xsmicd(NUM_GRP,NISO), xsmica(NUM_GRP,NISO), xsmics(NUM_GRP,NUM_GRP,NISO)
        real(XS_PREC)             ::  xdpmica(NUM_GRP,NISO), xdmmica(NUM_GRP,3,NISO), xddmica(NUM_GRP,NISO), &
                                xdpmicd(NUM_GRP,NISO), xdmmicd(NUM_GRP,3,NISO), xddmicd(NUM_GRP,NISO), &
                                xdpmics(NUM_GRP,NUM_GRP,NISO), xdmmics(NUM_GRP,NUM_GRP,3,NISO), xddmics(NUM_GRP,NUM_GRP,NISO)
        integer             :: ig, igs, ige   
        do ig=1,NUM_GRP
            xsmica(ig,ID_STRM)=this%rrsig(1,ig,3)
            xsmicd(ig,ID_STRM)=this%rrsig(1,ig,4)
        enddo
            
        do igs=1,NUM_GRP
        do ige=1,NUM_GRP
            if(igs.eq.ige) cycle
            xsmics(igs,ige,ID_STRM)=this%rrsigs(1,igs,ige,2)
        enddo
        enddo
        
        do ig=1,NUM_GRP
            xdpmica(ig,ID_STRM)=this%rrsig(2,ig,3)
            xdpmicd(ig,ID_STRM)=this%rrsig(2,ig,4)
            xdmmica(ig,1,ID_STRM)=this%rrsig(3,ig,3)
            xdmmicd(ig,1,ID_STRM)=this%rrsig(3,ig,4)
            xddmica(ig,ID_STRM)=this%rrsig(4,ig,3)
            xddmicd(ig,ID_STRM)=this%rrsig(4,ig,4)
        enddo
        
        do igs=1,NUM_GRP
        do ige=1,NUM_GRP
            if(igs.eq.ige) cycle
            xdpmics(igs,ige,ID_STRM)=this%rrsigs(2,igs,ige,2)
        enddo
        enddo
        
    end subroutine       
    
    subroutine calculateCorner(this, xsmica, xsmicd, xsmics, xdpmica, xdmmica, xddmica, &
                                                           xdpmicd, xdmmicd, xddmicd, &
                                                           xdpmics, xdmmics, xddmics )
        class(Reflector)    :: this
        real(XS_PREC)             ::  xsmicd(NUM_GRP,NISO), xsmica(NUM_GRP,NISO), xsmics(NUM_GRP,NUM_GRP,NISO)
        real(XS_PREC)             ::  xdpmica(NUM_GRP,NISO), xdmmica(NUM_GRP,3,NISO), xddmica(NUM_GRP,NISO), &
                                xdpmicd(NUM_GRP,NISO), xdmmicd(NUM_GRP,3,NISO), xddmicd(NUM_GRP,NISO), &
                                xdpmics(NUM_GRP,NUM_GRP,NISO), xdmmics(NUM_GRP,NUM_GRP,3,NISO), xddmics(NUM_GRP,NUM_GRP,NISO)
        integer             :: ig, igs, ige        
        do ig=1,NUM_GRP
            xsmica(ig,ID_STRM)=this%rrsig(1,ig,1)
            xsmicd(ig,ID_STRM)=this%rrsig(1,ig,2)
        enddo
            
        do igs=1,NUM_GRP
        do ige=1,NUM_GRP
            if(igs == ige) cycle
            xsmics(igs,ige,ID_STRM)=this%rrsigs(1,igs,ige,1)
        enddo
        enddo
        
        do ig=1,NUM_GRP
            xdpmica(ig,ID_STRM)=this%rrsig(2,ig,1)
            xdpmicd(ig,ID_STRM)=this%rrsig(2,ig,2)
            xdmmica(ig,1,ID_STRM)=this%rrsig(3,ig,1)
            xdmmicd(ig,1,ID_STRM)=this%rrsig(3,ig,2)
            xddmica(ig,ID_STRM)=this%rrsig(4,ig,1)
            xddmicd(ig,ID_STRM)=this%rrsig(4,ig,2)
        enddo
        
        do igs=1,NUM_GRP
        do ige=1,NUM_GRP
            if(igs.eq.ige) cycle
            xdpmics(igs,ige,ID_STRM)=this%rrsigs(2,igs,ige,1)
        enddo
        enddo
        
        
    end subroutine      

end module
