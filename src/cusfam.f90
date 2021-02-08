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
        
        call reset(stable, press)
    end subroutine
    
    subroutine getTHSatTemperature(tm)  bind(C, name="getTHSatTemperature")
        real(4)    :: tm
        tm = getSatTemperature(stable)
        
    end subroutine
    subroutine getTHDensity(h, dm)  bind(C, name="getTHDensity")
        real(4)    :: h, dm
        dm = getDensity(stable, h)
    end subroutine
    
    subroutine getTHTemperature(h, tm)  bind(C, name="getTHTemperature")
        real(4)    :: h, tm
        tm = getTemperature(stable, h)
    end subroutine

    subroutine getTHEnthalpy(tm, h)  bind(C, name="getTHEnthalpy")
        real(4)    :: h, tm
        h = getEnthalpy(stable, tm)
    end subroutine
    
    subroutine getTHCheckEnthalpy(h, err)  bind(C, name="checkTHEnthalpy")
        real(4)    :: h
        integer     :: err
        err = checkEnthalpy(stable, h)
    end subroutine

    
    
end module
