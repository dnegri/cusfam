module Icusfam
    use iso_c_binding
    implicit none
    
    
interface
    subroutine driveNodal() bind(C,name="driveNodalC")
    use iso_c_binding
    end subroutine
end interface
    
    
end module
