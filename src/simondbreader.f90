    module simondbreader

    integer :: ifile
    integer :: ng, nxyz 
contains
    subroutine opendb(lenf, file) bind(c, name="opendb")
        use, intrinsic :: iso_c_binding, only: c_char
        integer            :: lenf
        character(len=1, kind=c_char)   :: file(lenf)
        character*(lenf)                :: file2

        do i=1,lenf
            file2(i:i)= file(i)
        enddo
        !print *, lenf, file
        ifile = 6845
        open(unit=ifile, file=file2, form='unformatted', status='old')
    end subroutine
    
    subroutine closedb()  bind(c, name="closedb")
        close(ifile)
    end subroutine
    
    subroutine readDimension(ng_, nxy, nz, nx, ny, nsurf)  bind(c, name="readDimension")
        integer         :: ng_, nxy, nz, nx, ny, nsurf
        
        read(ifile) ng_, nxy, nz, nx, ny, nsurf
        nxyz = nxy*nz
        ng = ng_
    end subroutine
    
    subroutine readIndex(nx, ny, nxy, nz, nxs, nxe, nys, nye, nodel, neibr, hmesh)   bind(c, name="readIndex")
        integer         :: nxy, nx, ny, nz
        integer         :: nxs(ny), nxe(ny), nys(nx), nye(nx), nodel(nx,ny), neibr(4,nxy)
        real(4)         :: hmesh(3,nxy,nz)
        
        
        read(ifile) nxs, nxe, nys, nye, nodel, neibr, hmesh
        
    end subroutine
    
    subroutine readBoundary(symopt, symang, albedo)   bind(c, name="readBoundary")
        integer         :: symopt, symang
        real(4)         :: albedo(2,3)
        
        
        read(ifile) symopt, symang, albedo
        
    end subroutine
    
    subroutine readstep(bucyc, buavg, efpd)   bind(c, name="readStep")
        real(4)         :: bucyc, buavg, efpd
        
        read(ifile) bucyc, buavg, efpd
        
    end subroutine
    
    subroutine readDensity(niso, dnst)   bind(c, name="readDensity")
        real(4)            :: dnst(niso,nxyz)

        do l = 1,nxyz
            read(ifile) dnst(:,l) 
        enddo
    
    end subroutine
    
    subroutine readnxny(nx, ny, val)   bind(c, name="readNXNY")
        real(4)            :: val(nx, ny)

        read(ifile) val
    
    end subroutine    
    
    subroutine readnxyz(nxyz, val)   bind(c, name="readNXYZ")
        real(4)            :: val(nxyz)

        read(ifile) val
    
    end subroutine        
    
    
    subroutine readxs(niso, xs)   bind(c, name="readXS")
        real(4)            :: xs(ng,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xs(:,:,l) 
        enddo
    
    end subroutine
    
    subroutine readxsd(niso, xsd)  bind(c, name="readXSD")
        real(4)            :: xsd(ng,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xsd(:,:,l) 
        enddo
    
    end subroutine    
    
    subroutine readxss(niso, xss) bind(c, name="readXSS")
        real(4)            :: xss(ng,ng,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xss(:,:,:,l) 
        enddo
    
    end subroutine    
    
    subroutine readxssd(niso, xssd) bind(c, name="readXSSD")
        real(4)            :: xssd(ng,ng,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xssd(:,:,:,l) 
        enddo
    
    end subroutine    
    
    subroutine readxsdtm(niso, xs) bind(c, name="readXSDTM")
        real(4)            :: xs(ng,2,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xs(:,:,:,l)
        enddo
    
    end subroutine
    
    
    subroutine readxssdtm(niso, xss)  bind(c, name="readXSSDTM")
        real(4)            :: xss(ng, ng,2,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xss(:,:,:,:,l)
        enddo
    
    end subroutine       
end module