module cusfam
    use iso_c_binding
    implicit none
    
    
interface
    subroutine initCudaGeometry(ng, nxy, nz, nx, ny, nxs, nxe, nys, nye, nsurf, ijtol, neibr, hmesh, symopt, symang, albedo) bind(C,name="initCudaGeometry")
        use iso_c_binding
        integer         :: symopt, symang
        real            :: albedo(*)
        integer         :: ng, nxy, nz, nx, ny, nsurf
        integer         :: nxs(ny), nxe(ny), nys(nx), nye(nx), neibr(4,nxy), ijtol(nx,ny)
        real            :: hmesh(3, nxy, nz) 
    end subroutine
    
    subroutine initCudaXS(ng, nxy, nz, xsdf, xstf, xsnf, xssf, xschif, xsadf) bind(C, name="initCudaXS")
        integer              :: ng, nxy, nz    
        real  :: xsdf(ng, nxy, nz), xstf(ng, nxy, nz), xsnf(ng, nxy, nz), &
                xssf(ng, ng, nxy, nz), xschif(ng, nxy, nz), xsadf(ng, nxy, nz)
    end subroutine
    
    subroutine initCudaSolver() bind(C, name="initCudaSolver")
    end subroutine
    subroutine updateCuda(reigv, jnet, phif) bind(C, name="updateCuda")
        real        :: reigv, jnet(*), phif(*)
    end subroutine
    
    subroutine runCuda(jnet) bind(C, name="runCuda")
    real        :: jnet(*)
    end subroutine
    
    subroutine runCMFD(reigv, psi, jnet, phif) bind(C, name="runCMFD")
        real        :: reigv, psi(*), jnet(*), phif(*)
    end subroutine
    
end interface


    
    
end module
