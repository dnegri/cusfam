module cusfam
    use iso_c_binding
    use CSteamTable
    implicit none
    
    type(SteamTable), pointer :: stable
contains
    
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
