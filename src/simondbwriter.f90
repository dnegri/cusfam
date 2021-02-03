    module simondbwriter

    implicit none



    integer :: ifile
    contains
    subroutine opendb(file)

    character*(*)   :: file

    ifile = 6845
    open(unit=ifile, file=file, form='unformatted', status='unknown')
    end subroutine

    subroutine closedb()
    close(ifile)
    end subroutine


    subroutine writegeom()
    use geom
    use const
    real(4)         :: hmesh4(NDIRMAX,nxy,nz), albedo4(2,NDIRMAX)

    hmesh4 = hmesh(1:NDIRMAX,:,:)
    albedo4 = albedo

    write(ifile) ng, nxy, nz, nx, ny, nsurf
    write(ifile) nxs, nxe, nys, nye, nodel, neibr, hmesh4
    write(ifile) 1, isymang, albedo4

    end subroutine

    subroutine writestep()
    use geom
    use var_dusinfo, only : bucoini, bucocyc, bucoavg, efpd

    real(4)         :: bucyc4, buavg4, efpd4

    bucyc4 = bucocyc
    buavg4 = bucoavg
    efpd4 = efpd
    write(ifile) bucyc4, buavg4, efpd4
    call writexsall()

    end subroutine

    subroutine writexsall()
    use var_dctl, only :  &
        nfcnt  &
        , iniso  &
        , infiso

    use var_pisoxsd, only :   &
        xdpmicd    &
        , xdpmica    &
        , xdpmics    &
        , xdmmicd    &
        , xdmmica    &
        , xdmmics    &
        , xddmicd    &
        , xddmica    &
        , xddmics    &
        , xdfmicd    &
        , xdfmica    &
        , xdfmics    &
        , xdpmicf    &
        , xdfmicf    &
        , xdmmicf    &
        , xddmicf    &
        , xdpmicn    &
        , xdfmicn    &
        , xdmmicn    &
        , xddmicn

    use var_pdusxs, only :    &
        xsmicd0    &
        , xsmica0    &
        , xsmics0    &
        , xsmicf0    &
        , xsmicn0    &
        , xsmick0    &
        , xsmicd &
        , xsmica &
        , xsmics &
        , xsmicf &
        , xsmicn

    use var_dusdim, only : nnis
    use var_micxs, only  :np, ndrv


    call writexs(ifile, xsmicd0)
    call writexs(ifile, xsmica0)
    call writexs(ifile, xsmicf0)
    call writexs(ifile, xsmick0)
    call writexs(ifile, xsmicn0)
    call writexss(ifile, xsmics0)

    call writexsd(ifile, xdfmicd)
    call writexsd(ifile, xddmicd)
    call writexsd(ifile, xdpmicd)
    call writexsd(ifile, xdpmicf)
    call writexsd(ifile, xdfmicf)
    call writexsd(ifile, xddmicf)
    call writexsd(ifile, xdfmica)
    call writexsd(ifile, xddmica)
    call writexsd(ifile, xdpmica)
    call writexsd(ifile, xdpmicn)
    call writexsd(ifile, xdfmicn)
    call writexsd(ifile, xddmicn)
    call writexssd(ifile, xdfmics)
    call writexssd(ifile, xddmics)
    call writexssd(ifile, xdpmics)

    call writexstm(ifile, xdmmicd)
    call writexstm(ifile, xdmmicf)
    call writexstm(ifile, xdmmica)
    call writexstm(ifile, xdmmicn)
    call writexsstm(ifile, xdmmics)
    end subroutine

    subroutine writexs(ifile, xs)
    use geom,only: nxy, nz
    integer         :: m2d, k
    integer         :: ifile
    real, pointer   :: xs(:,:,:,:)

    do k = 1,nz
        do m2d = 1,nxy
            write(ifile) real(xs(:,m2d,k,:),4)
        enddo
    enddo

    end subroutine

    subroutine writexsd(ifile, xsd)
    use geom,only: nxy, nz
    integer         :: m2d, k
    integer         :: ifile
    real, pointer   :: xsd(:,:,:,:,:)

    do k = 1,nz
        do m2d = 1,nxy
            write(ifile) real(xsd(:,1,m2d,k,:),4)
        enddo
    enddo

    end subroutine

    subroutine writexss(ifile, xss)
    use geom,only: nxy, nz
    integer         :: m2d, k
    integer         :: ifile
    real, pointer   :: xss(:,:,:,:,:)

    do k = 1,nz
        do m2d = 1,nxy
            write(ifile) real(xss(:,:,m2d,k,:) ,4)
        enddo
    enddo

    end subroutine

    subroutine writexssd(ifile, xssd)
    use geom,only: nxy, nz
    integer         :: m2d, k
    integer         :: ifile
    real, pointer   :: xssd(:,:,:,:,:,:)

    do k = 1,nz
        do m2d = 1,nxy
            write(ifile) real(xssd(:,:,1,m2d,k,:) ,4)
        enddo
    enddo

    end subroutine

    subroutine writexstm(ifile, xs)
    use geom,only: nxy, nz
    integer         :: m2d, k
    integer         :: ifile
    real, pointer   :: xs(:,:,:,:,:)

    do k = 1,nz
        do m2d = 1,nxy
            write(ifile) real(xs(:,1:2,m2d,k,:) , 4)
        enddo
    enddo

    end subroutine


    subroutine writexsstm(ifile, xss)
    use geom,only: nxy, nz
    integer         :: m2d, k
    integer         :: ifile
    real, pointer   :: xss(:,:,:,:,:,:)

    do k = 1,nz
        do m2d = 1,nxy
            write(ifile) real(xss(:,:,1:2,m2d,k,:) ,4)
        enddo
    enddo

    end subroutine
    end module