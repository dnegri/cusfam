module cusfam
    use iso_c_binding
    implicit none
    
    
interface
    subroutine setBoundaryCondtition(symopt, symang, albedo) bind(C,name="setBoundaryCondtition")
        use iso_c_binding
        integer         :: symopt, symang
        real            :: albedo(*)
    end subroutine

    subroutine initGeometry(ng, nxy, nz, nx, ny, nxs, nxe, nys, nye, nsurf, ijtol, neibr, hmesh) bind(C,name="initGeometry")
        use iso_c_binding
        integer         :: ng, nxy, nz, nx, ny, nsurf
        integer         :: nxs(ny), nxe(ny), nys(nx), nye(nx), neibr(4,nxy), ijtol(nx,ny)
        real            :: hmesh(3+1, nxy, nz) !hmesh has 0-th index 
    end subroutine
    
    subroutine initCrossSection(ng, nxy, nz, xsdf, xstf, xsnf, xssf, xschif, xsadf) bind(C, name="initCrossSection")
        integer              :: ng, nxy, nz    
        real  :: xsdf(ng, nxy, nz), xstf(ng, nxy, nz), xsnf(ng, nxy, nz), &
                xssf(ng, ng, nxy, nz), xschif(ng, nxy, nz), xsadf(ng, nxy, nz)
    end subroutine
    
    subroutine initSANM2N() bind(C, name="initSANM2N")
    end subroutine
    subroutine resetSANM2N(reigv, jnet, phif) bind(C, name="resetSANM2N")
        real        :: reigv, jnet(*), phif(*)
    end subroutine
    
    subroutine runSANM2N() bind(C, name="runSANM2N")
    end subroutine
end interface
    
    
end module
