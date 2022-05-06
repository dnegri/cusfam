module CFormFunction
    use iso_c_binding
    implicit none
    private
                                                               
    character*(*), parameter    :: EORCOM =' *EOR'
    character*(*), parameter    :: FORMAT_HEADER = "(A4,A44,I1,I3,I1,1X,I2,I1,1X,I2,3I4)"
    integer, parameter   :: NUM_GRP     = 2
    integer, parameter   :: NUM_CELLXY  = 16
    integer, parameter   :: LEN_COMPNAME = 13

    
    type, private   :: CompFormFunction 
        integer            :: ihff
        integer            :: nburn
        real(4), pointer   :: burn(:)
        real(4), pointer   :: hff(:,:,:)
    contains
        procedure   init => initCFF
    end type
    
    type, public   :: FormFunction
        integer                                 :: nhff, ng, ncellxy
        character(len=LEN_COMPNAME), pointer    :: hffnames(:)
        
        type(CompFormFunction), pointer     :: cff(:)
        
    contains
        procedure init
        procedure readFile
        procedure indexOfFormFunction
    end type
    
    
    public :: readFormFunction, getFormFunction
    
contains

    subroutine initCFF(this, mxburn, ihff, ncellxy)
        class(CompFormFunction) :: this
        integer                 :: mxburn, ihff, ncellxy
        
        this%ihff = ihff
        this%nburn = 0
        
        allocate(this%burn(mxburn))
        allocate(this%hff(ncellxy, ncellxy, mxburn))

        
    end subroutine
    
    subroutine init(this, nhff, hffnames, ng, ncellxy)
        class(FormFunction) :: this
        integer             :: nhff, ng, ncellxy
        character*(LEN_COMPNAME)   :: hffnames(nhff)
        
        integer         :: i
        
        this%nhff = nhff
        this%ng = ng
        this%ncellxy = ncellxy
        
        allocate(this%hffnames(nhff))
        allocate(this%cff(nhff))
        
        do i = 1, nhff
            call this%cff(i)%init(100, i, ncellxy)
            this%hffnames(i) = hffnames(i)
        enddo
            
    end subroutine

    
    subroutine readFile(this, filename)
        class(FormFunction) :: this
        character*(*)       :: filename
        character*(80)              :: header
        character*(LEN_COMPNAME)    :: hffname
        class(CompFormFunction), pointer   :: cff
        integer         :: ifile, ihff, jp, ip, ig
        real            :: burn
        
        ifile = 1002
        open(ifile,file=filename,status='old')
        
        do while(.true.)
            read (ifile,'(a)',end=1,err=2) header

            read(ifile,'(a13)') hffname
            read(ifile,*) burn
        
            ihff = this%indexOfFormFunction(hffname)
            if(ihff .ne. 0) then
                cff => this%cff(ihff)
                cff%nburn = cff%nburn+1
                cff%burn(cff%nburn) = burn
                do jp = 1, this%ncellxy
                    read(ifile, *) (cff%hff(ip,jp,cff%nburn), ip=1, this%ncellxy)
                enddo
            else
                do jp = 1, this%ncellxy
                    read(ifile, *) header
                enddo
            endif

            do jp = 1, this%ncellxy
                read(ifile, *)
            enddo
            
            do ig=1, this%ng
            do jp = 1, this%ncellxy
                read(ifile, *) header
            enddo
            enddo
            
            read(ifile, *) header
            
            if(ihff .eq. 0) cycle
            
        enddo
         
1       continue
2       continue
        close(ifile)
    end subroutine    

    
    function indexOfFormFunction(this, hffname) result(ihff)
        class(FormFunction)     :: this
        character*(*)       :: hffname
        integer             :: ihff, i
        
        ihff = 0
        do i=1,this%nhff
            if(this%hffnames(i) == hffname) then
                ihff = i
                exit
            endif
        enddo
    end function
    
    subroutine getFormFunction(ff_ptr, burn, ihff, hff) bind(c, name="getFormFunction")
        type(c_ptr), INTENT(IN), value:: ff_ptr    
        integer             :: ihff
        real(XS_PREC)       :: burn
        real(XS_PREC)       :: hff(NUM_CELLXY, NUM_CELLXY)
        class(CompFormFunction), pointer   :: cff
        integer                     :: klo, jp, ip, ig, ihffref
        real(XS_PREC)               :: af(3)
        type(FormFunction), pointer   :: this
        real(XS_PREC), pointer  :: hff_ptr
        character*(LEN_COMPNAME)    :: hffname
        
        call c_f_pointer(ff_ptr, this)
        
        hffname = this.hffnames(ihff)
        ihffref = this%indexOfFormFunction(hffname)
        
        cff => this%cff(ihffref)
        
        
        call quad1(cff%nburn, cff%burn(:), burn, klo, af)

        do jp=1,this%ncellxy
        do ip=1,this%ncellxy
            hff(ip,jp) = af(1)*cff%hff(ip,jp,klo) + af(2)*cff%hff(ip,jp,klo+1) + af(3)*cff%hff(ip,jp,klo+2)
        enddo
        enddo
    end subroutine
    

    function readFormFunction(lenf, file, nhff, hffnames) result(ff_ptr) bind(c, name="readFormFunction")
        use iso_c_binding, only: c_ptr, c_int, c_f_pointer, c_loc, c_null_char
        integer                         :: lenf
        character(len=1, kind=c_char)   :: file(lenf)
        character*(lenf)                :: file2
        
        integer(kind=c_int),                 intent(in) :: nhff
        type(c_ptr), target,                 intent(in) :: hffnames
        type(c_ptr)                                     :: ff_ptr
        character(kind=c_char), dimension(:,:), pointer :: fptr
        character(len=13), dimension(nhff)             :: fstring
        integer                                         :: i, slen
        
        type(FormFunction), pointer :: ff
        
        fstring(:) = "            "
        
        call c_f_pointer(c_loc(hffnames), fptr, [13, nhff])
        do i = 1, nhff
            slen = 0
            do while(fptr(slen+1,i) /= c_null_char)
                slen = slen + 1
            end do
            fstring(i) = transfer(fptr(1:slen,i), fstring(i)(1:slen))
        enddo
        
        do i=1,lenf
            file2(i:i)= file(i)
        enddo
        
        allocate(ff)
        
        call ff%init(nhff, fstring, NUM_GRP, NUM_CELLXY)
        call ff%readFile(file2)
        
        ff_ptr = c_loc(ff)
    end function
    
     
    
end module
