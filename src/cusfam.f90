module cusfam
    use iso_c_binding
    use CSteamTable
    use CTableSet
    implicit none
    
    type(SteamTable), pointer :: stable
    type(TableSet)            :: tset
    integer                   :: ng =2, NISO = 40
contains

    subroutine readTableSet(lenf, file, ncomp, compnames) bind(c, name="readTableSet")
        use iso_c_binding, only: c_ptr, c_int, c_f_pointer, c_loc, c_null_char
        integer                         :: lenf
        character(len=1, kind=c_char)   :: file(lenf)
        character*(lenf)                :: file2
        
        integer(kind=c_int),                 intent(in) :: ncomp
        type(c_ptr), target,                 intent(in) :: compnames
        character(kind=c_char), dimension(:,:), pointer :: fptr
        character(len=13), dimension(ncomp)             :: fstring
        integer                                         :: i, slen
        
        fstring(:) = "            "
        
        call c_f_pointer(c_loc(compnames), fptr, [13, ncomp])
        do i = 1, ncomp
            slen = 0
            do while(fptr(slen+1,i) /= c_null_char)
                slen = slen + 1
            end do
            fstring(i) = transfer(fptr(1:slen,i), fstring(i)(1:slen))
        enddo
        
        do i=1,lenf
            file2(i:i)= file(i)
        enddo
        
        call tset%init(ncomp, fstring)
        call tset%readFile(file2)
    end subroutine
    
    subroutine calculateReference(icomp, burn, xsmicd, xsmica, xsmicn, xsmicf, xsmick, xsmics, xsmic2n, xehfp) bind(c, name="calculateReference")
        integer             :: icomp
        real(4)             :: burn
        real(4)             :: xsmicd(ng,NISO), xsmica(ng,NISO), xsmicn(ng,NISO), xsmicf(ng,NISO), xsmick(ng,NISO), xsmics(ng,ng,NISO), xsmic2n(ng), xehfp

        call tset%comps(icomp)%calculateReference(burn, xsmicd, xsmica, xsmicn, xsmicf, xsmick, xsmics, xsmic2n, xehfp)
        
    end subroutine
    
    subroutine calculateVariation(icomp, burn,   xdpmicn, xdfmicn, xdmmicn, xddmicn, &
                                                xdpmicf, xdfmicf, xdmmicf, xddmicf, &
                                                xdpmica, xdfmica, xdmmica, xddmica, &
                                                xdpmicd, xdfmicd, xdmmicd, xddmicd, &
                                                xdpmics, xdfmics, xdmmics, xddmics )  bind(c, name="calculateVariation")
        integer             :: icomp
        real(4)             :: burn
        real(4)             ::  xdpmicn(ng,NISO), xdfmicn(ng,NISO), xdmmicn(ng,3,NISO), xddmicn(ng,NISO), &
                                xdpmicf(ng,NISO), xdfmicf(ng,NISO), xdmmicf(ng,3,NISO), xddmicf(ng,NISO), &
                                xdpmica(ng,NISO), xdfmica(ng,NISO), xdmmica(ng,3,NISO), xddmica(ng,NISO), &
                                xdpmicd(ng,NISO), xdfmicd(ng,NISO), xdmmicd(ng,3,NISO), xddmicd(ng,NISO), &
                                xdpmics(ng,ng,NISO), xdfmics(ng,ng,NISO), xdmmics(ng,ng,3,NISO), xddmics(ng,ng,NISO)
        
        call tset%comps(icomp)%calculateVariation(burn, xdpmicn, xdfmicn, xdmmicn, xddmicn, &
                                                xdpmicf, xdfmicf, xdmmicf, xddmicf, &
                                                xdpmica, xdfmica, xdmmica, xddmica, &
                                                xdpmicd, xdfmicd, xdmmicd, xddmicd, &
                                                xdpmics, xdfmics, xdmmics, xddmics )
        
    end subroutine       


    subroutine setTHPressure(press)   bind(C, name="setTHPressure")
        real(4)    :: press
        if(.not.associated(stable)) then
            allocate(stable)
        endif
        
        call stable%reset(dble(press))
    end subroutine
    
    subroutine getTHSatTemperature(tm)  bind(C, name="getTHSatTemperature")
        real(4)    :: tm
        tm = stable%getSatTemperature()
        
    end subroutine
    subroutine getTHDensity(h, dm)  bind(C, name="getTHDensity")
        real(4)    :: h, dm
        dm = stable%getDensity(dble(h))
    end subroutine
    
    subroutine getTHTemperature(h, tm)  bind(C, name="getTHTemperature")
        real(4)    :: h, tm
        tm = stable%getTemperature(dble(h))
    end subroutine

    subroutine getTHEnthalpy(tm, h)  bind(C, name="getTHEnthalpy")
        real(4)    :: h, tm
        h = stable%getEnthalpy(dble(tm))
    end subroutine
    
    subroutine getTHCheckEnthalpy(h, err)  bind(C, name="checkTHEnthalpy")
        real(4)    :: h
        integer     :: err
        err = stable%checkEnthalpy(dble(h))
    end subroutine

    subroutine copySteamTable(press &				 
              ,  tmin   &
              ,  tmax   &
              ,  dmin   &
              ,  dmax   &
              ,  dgas   &
              ,  hmin   &
              ,  hmax   &
              ,  hgas   &
              ,  vismin   &
              ,  vismax   &
              ,  visgas   &
              ,  tcmin   &
              ,  tcmax   &
              ,  tcgas   &
              ,  shmin   &
              ,  shmax   &
              ,  shgas   &
              ,  rhdel  &
              ,  rtdel  &
              ,  rhdiff, cmn, propc, hmod, dmodref) bind(C, name="copySteamTable")
    
    real(4)     	 press, tmin   &
                  ,  tmax   &
                  ,  dmin   &
                  ,  dmax   &
                  ,  dgas   &
                  ,  hmin   &
                  ,  hmax   &
                  ,  hgas   &
                  ,  vismin   &
                  ,  vismax   &
                  ,  visgas   &
                  ,  tcmin   &
                  ,  tcmax   &
                  ,  tcgas   &
                  ,  shmin   &
                  ,  shmax   &
                  ,  shgas   &
                  ,  rhdel  &
                  ,  rtdel  &
                  ,  rhdiff

      real(4)     :: cmn(0:3,2)                 &
                  ,  propc(NPROP,2,100)     &
                  ,  hmod(2,100)            &
                  ,  dmodref(2,100)

        call setTHPressure(press)
        
        tmin	= stable%tmin
        tmax	= stable%tmax
        dmin	= stable%dmin
        dmax	= stable%dmax
        dgas	= stable%dgas
        hmin	= stable%hmin
        hmax	= stable%hmax
        hgas	= stable%hgas
        vismin	= stable%vismin
        vismax	= stable%vismax
        visgas	= stable%visgas
        tcmin	= stable%tcmin
        tcmax	= stable%tcmax
        tcgas	= stable%tcgas
        shmin	= stable%shmin
        shmax	= stable%shmax
        shgas	= stable%shgas
        rhdel	= stable%rhdel
        rtdel	= stable%rtdel
        rhdiff	= stable%rhdiff     
        propc = stable%propc(:,:,1:100)
        hmod = stable%hmod(:,1:100)
        dmodref = stable%dmodref(:,1:100)
        cmn = stable%cmn(:,:)
    end subroutine
    
end module
