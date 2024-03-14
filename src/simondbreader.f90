    module simondbreader

    integer :: ifile
    integer, parameter :: ngrp = 2
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
    end subroutine
    
    subroutine readIndex(nx, ny, nxy, nz, nxs, nxe, nys, nye, nodel, rotflg, neibr, hmesh)   bind(c, name="readIndex")
        integer         :: nxy, nx, ny, nz
        integer         :: nxs(ny), nxe(ny), nys(nx), nye(nx), nodel(nx,ny), rotflg(nx,ny),neibr(4,nxy)
        real(4)         :: hmesh(3,nxy,nz)
        
        
        read(ifile) nxs, nxe, nys, nye, nodel, rotflg, neibr, hmesh
        
    end subroutine
    
    subroutine readComposition(nxy, nz, ncomp, names, comps)   bind(c, name="readComposition")
        use iso_c_binding, only: c_ptr, c_int, c_f_pointer, c_loc, c_null_char , c_char   
        integer                                 :: nxy, nz, ncomp
        integer                                 :: comps(nxy,nz)
        integer                                 :: b10wp0(nxy,nz)
        type(c_ptr),target                      :: names
        character(len=12), dimension(50)        :: fnames
        character(kind=c_char), dimension(:,:), pointer :: fptr
        integer                             :: i, j, k, l
        
        read(ifile) ncomp
        read(ifile) fnames(1:ncomp)
        read(ifile) comps
        
        call c_f_pointer(c_loc(names), fptr, [13, 50])
        do i = 1, ncomp
            do j=1, 12
                fptr(j,i) = fnames(i)(j:j)
            enddo
            fptr(13,i) = c_null_char
        enddo
                    
    end subroutine
    
    subroutine readString(n, length, strings)   bind(c, name="readString")
        use iso_c_binding, only: c_ptr, c_int, c_f_pointer, c_loc, c_null_char , c_char   
        integer                                 :: n
        type(c_ptr),target                      :: strings
        character(len=length-1), dimension(n)     :: fstrings
        character(kind=c_char), dimension(:,:), pointer :: fptr
        integer                             :: i, j, k, l
        
        read(ifile) fstrings(1:n)
        
        call c_f_pointer(c_loc(strings), fptr, [length, n])
        do i = 1, n
            do j=1, length-1
                fptr(j,i) = fstrings(i)(j:j)
            enddo
            fptr(length,i) = c_null_char
        enddo
                    
    end subroutine
    
    
    subroutine readConstantI(n, cnst)   bind(c, name="readConstantI")
        integer         :: n
        integer         :: cnst(n)
        read(ifile) cnst(1:n)
        
    end subroutine
    
    subroutine readConstantF(n, cnst)   bind(c, name="readConstantF")
        integer         :: n
        real(4)         :: cnst(n)
        
        read(ifile) cnst(1:n)
        
    end subroutine

    subroutine writeConstantF(n, cnst)   bind(c, name="writeConstantF")
        integer         :: n
        real(4)         :: cnst(n)
        
        write(ifile) cnst(1:n)
        
    end subroutine
    
    subroutine readConstantD(n, cnst)   bind(c, name="readConstantD")
        integer         :: n
        real(8)         :: cnst(n)
        
        read(ifile) cnst(1:n)
        
    end subroutine
    
    subroutine writeConstantD(n, cnst)   bind(c, name="writeConstantD")
        integer         :: n
        real(8)         :: cnst(n)
        
        write(ifile) cnst(1:n)
        
    end subroutine
    
    subroutine readBoundary(symopt, symang, albedo)   bind(c, name="readBoundary")
        integer         :: symopt, symang
        real(4)         :: albedo(2,3)
        
        
        read(ifile) symopt, symang, albedo
        
    end subroutine
    
    subroutine readstep(power, bucyc, buavg, efpd, eigvc, eigv, fnorm)   bind(c, name="readStep")
        real(4)         :: power, bucyc, buavg, efpd
        real(8)         :: eigvc, eigv, fnorm
        
        read(ifile) power
        read(ifile) bucyc
        read(ifile) buavg
        read(ifile) efpd
        read(ifile) eigvc
        read(ifile) eigv
        read(ifile) fnorm
        
    end subroutine
    
    subroutine writestep(power, bucyc, buavg, efpd, eigvc, eigv, fnorm)   bind(c, name="writeStep")
        real(4)         :: power, bucyc, buavg, efpd
        real(8)         :: eigv, fnorm
        
        write(ifile) power
        write(ifile) bucyc
        write(ifile) buavg
        write(ifile) efpd
        write(ifile) eigvc
        write(ifile) eigv
        write(ifile) fnorm
        
    end subroutine
    
    subroutine readDensity(niso, nxyz, dnst)   bind(c, name="readDensity")
        real(4)            :: dnst(niso,nxyz)

        do l = 1,nxyz
            read(ifile) dnst(:,l) 
        enddo
    
    end subroutine
    
    subroutine writeDensity(niso, nxyz, dnst)   bind(c, name="writeDensity")
        real(4)            :: dnst(niso,nxyz)

        do l = 1,nxyz
            write(ifile) dnst(:,l) 
        enddo
    
    end subroutine
    
    subroutine readnxny(nx, ny, val)   bind(c, name="readNXNY")
        real(4)            :: val(nx, ny)

        read(ifile) val
    
    end subroutine    

    subroutine readnxyzi(nxyz, val)   bind(c, name="readNXYZI")
        integer            :: val(nxyz)

        read(ifile) val
    
    end subroutine        
    
    subroutine readnxyz(nxyz, val)   bind(c, name="readNXYZ")
        real(4)            :: val(nxyz)

        read(ifile) val
    
    end subroutine        
    
    subroutine readnxyz8(nxyz, val)   bind(c, name="readNXYZ8")
        real(8)            :: val(nxyz)

        read(ifile) val
    
    end subroutine        
    
    
    subroutine readxs(niso, nxyz, xs)   bind(c, name="readXS")
        real(4)            :: xs(ngrp,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xs(:,:,l) 
        enddo
    
    end subroutine
    
    subroutine readxsd(niso, nxyz, xsd)  bind(c, name="readXSD")
        real(4)            :: xsd(ngrp,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xsd(:,:,l) 
        enddo
    
    end subroutine    
    
    subroutine readxss(niso, nxyz, xss) bind(c, name="readXSS")
        real(4)            :: xss(ngrp,ngrp,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xss(:,:,:,l) 
        enddo
    
    end subroutine    
    
    subroutine readxssd(niso, nxyz, xssd) bind(c, name="readXSSD")
        real(4)            :: xssd(ngrp,ngrp,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xssd(:,:,:,l) 
        enddo
    
    end subroutine    
    
    subroutine readxsdtm(niso, nxyz, xs) bind(c, name="readXSDTM")
        real(4)            :: xs(ngrp,3,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xs(:,:,:,l)
        enddo
    
    end subroutine
    
    
    subroutine readxssdtm(niso, nxyz, xss)  bind(c, name="readXSSDTM")
        real(4)            :: xss(ngrp, ngrp,3,niso,nxyz)

        do l = 1,nxyz
            read(ifile) xss(:,:,:,:,l)
        enddo
    
    end subroutine       
end module