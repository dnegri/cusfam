module simondbreader

    integer :: ifile
    integer :: ng, nxyz 
    integer :: nnucl = 40
contains
    subroutine opendb(file)

        character*(*)   :: file
    
        ifile = 6845
        open(unit=ifile, file=file, form='unformatted', status='old')
    end subroutine
    
    subroutine closedb()    
        close(ifile)
    end subroutine
    
    subroutine readgeom(ifile, ng, nxy, nz, nx, ny, nsurf)
        use geom
        use const
        integer         :: ifile
        integer         :: ng, nxy, nz, nx, ny, nsurf
        
        read(ifile) ng, nxy, nz, nx, ny, nsurf
    end subroutine
    
    subroutine readgeom(ifile, ng, nxy, nz, nx, ny, nsurf, nxs, nxe, nys, nye, nodel, neibr, hmesh, symopt, symang, albedo)
        integer         :: ifile
        integer         :: ng, nxy, nz, nx, ny, nsurf
        integer         :: nxs(ny), nxe(ny), nys(nx), nye(nx), nodel(nx,ny), neibr(4,nxy)
        integer         :: symopt, symang
        real(4)         :: hmesh(3,nxy,nz), albedo(2,3)
        
        
        read(ifile) ng, nxy, nz, nx, ny, nsurf
        read(ifile) nxs, nxe, nys, nye, nodel, neibr, hmesh
        read(ifile) symopt, symang, albedo
        
    end subroutine
    
    subroutine readstep(ifile, bucyc, buavg, efpd)
        
        integer         :: ifile
        real(4)         :: bucyc, buavg, efpd
        
        read(ifile) bucyc, buavg, efpd
        
    end subroutine
    
    subroutine readxs(ifile, xs)
        integer         :: m2d, k
        integer         :: ifile
        real            :: xs(ig,nnucl,nxyz)

        do l = 1,nxyz
            read(ifile) xs(:,:,m2d,k) 
        enddo
    
    end subroutine
    
    subroutine readxsd(ifile, xsd)
        integer         :: m2d, k
        integer         :: ifile
        real            :: xsd(ig,nnucl,nxyz)

        do l = 1,nxyz
            read(ifile) xsd(:,:,l) 
        enddo
    
    end subroutine    
    
    subroutine readxss(ifile, xss)
        integer         :: m2d, k
        integer         :: ifile
        real, pointer   :: xss(ng,ng,nnucl,nxyz)

        do l = 1,nxyz
            read(ifile) xss(:,:,:,l) 
        enddo
    
    end subroutine    
    
    subroutine readxssd(ifile, xssd)
        integer         :: m2d, k
        integer         :: ifile
        real            :: xssd(ng,ng,nnucl,nxyz)

        do l = 1,nxyz
            read(ifile) xssd(:,:,:,l) 
        enddo
    
    end subroutine    
    
    subroutine readxstm(ifile, xs)
        integer         :: m2d, k
        integer         :: ifile
        real, pointer   :: xs(ng,2,nnucl,nxyz)

        do l = 1,nxyz
            read(ifile) xs(:,:,:,l)
        enddo
    
    end subroutine
    
    
    subroutine readxsstm(ifile, xss)
        integer         :: m2d, k
        integer         :: ifile
        real, pointer   :: xss(ng, ng,2,nnucl,nxyz)

        do l = 1,nxyz
            read(ifile) xss(:,:,:,:,l)
        enddo
    
    end subroutine       
end module