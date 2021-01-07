module Icusfam
    use iso_c_binding
    implicit none
    
    
interface
    subroutine initNodal(ng, nxy, nz, nsurf, neibr, hmesh) bind(C,name="initNodal")
    use iso_c_binding
    integer         :: ng, nxy, nz, nsurf
    integer         :: neibr(*)
    real(8)         :: hmesh(*)
    end subroutine
end interface
    
    
end module
