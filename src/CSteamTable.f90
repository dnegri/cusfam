!**********************************************************************!
!                                                                      !
! KEPCO Nuclear Fuel CONFIDENTIAL                                      !
! __________________                                                   !
!                                                                      !
!  [2007] - [CURRENT] KEPCO Nuclear Fuel Incorporated                  !
!  All Rights Reserved.                                                !
!                                                                      !
! NOTICE:  All information contained herein is, and remains the        !
!          property of KEPCO Nuclear Fuel Incorporated and its         !
!          suppliers, if any. The intellectual and technical           !
!          concepts contained herein are proprietary to KEPCO          !
!          Nuclear Fuel Incorporated and protected by trade            !
!          secret or copyright law.                                    !
!          Dissemination of this information or reproduction of        !
!          this material is strictly forbidden unless prior            !
!          written permission is obtained from KEPCO Nuclear           !
!          Fuel Incorporated.                                          !
!**********************************************************************!

!**********************************************************************!
! Revision History :                                                   !
! 1.0.0  - Initial Release (Jooil Yoon)                                !
!**********************************************************************!

module CSteamTable

   implicit none
   private

   real, parameter   :: psia2bar1 = 273.15
   real, parameter   :: psia2bar    = 6.894757E-02,   &
           ft3lbm2gcc  = 1.601846E-02,   &
           btulbm2kjkg = 2.326,          &
           bar2psia    = 1./psia2bar,    &
           bar2mpa     = 0.1,            &
           kjkg2btulbm = 1./btulbm2kjkg, &
           lbm2g       = 453.59237,      &
           ft2cm       = 30.48,          &
           btu2j       = 1055.056,       &
           btuhrft22wm2= btu2j/(ft2cm**2*0.3600),   &
           rbarn       = 1.0E-24
   real   , parameter :: KELVIN=273.15,PI=3.1415927
   real, parameter                  :: RMILLI = 1.E-3

   integer, public, parameter   :: STEAM_TABLE_ERROR_NO      = 0
   integer, public, parameter   :: STEAM_TABLE_ERROR_MAXENTH = 1
   integer, public, parameter   :: STEAM_TABLE_ERROR_MINENTH = 2
   integer, public, parameter   :: STEAM_TABLE_ERROR_MAXTEMP = 3
   integer, public, parameter   :: STEAM_TABLE_ERROR_MINTEMP = 4

   integer, parameter         :: NPROP = 6
   integer, parameter         :: PROP_TEMP = 1
   integer, parameter         :: PROP_ENTH = 2
   integer, parameter         :: PROP_DENS = 3
   integer, parameter         :: PROP_VISC = 4
   integer, parameter         :: PROP_TCON = 5
   integer, parameter         :: PROP_SPCH = 6

   integer                      :: nSteamTablePoints = 100

   type, public   :: SteamTable
      private

      integer        :: npnts

      real           :: press = 0.0

      real           :: tmin  = 20.0   &
              ,  tmax  = 0.0    &
              ,  dmin  = 0.0    &
              ,  dmax  = 0.0    &
              ,  dgas  = 0.0    &
              ,  hmin  = 0.0    &
              ,  hmax  = 0.0    &
              ,  hgas  = 0.0    &
              ,  vismin  = 0.0    &
              ,  vismax  = 0.0    &
              ,  visgas  = 0.0    &
              ,  tcmin  = 0.0    &
              ,  tcmax  = 0.0    &
              ,  tcgas  = 0.0    &
              ,  shmin  = 0.0    &
              ,  shmax  = 0.0    &
              ,  shgas  = 0.0    &
              ,  rhdel = 0.0    &
              ,  rtdel = 0.0    &
              ,  rhdiff= 0.0

      real           :: cmn(0:3,2)

      real, pointer  :: propc(:,:,:)      &
              ,  hmod(:,:)      &
              ,  dmodref(:,:)

   contains
      procedure reset
      procedure destroy
      procedure getPressure
      procedure getEnthalpy
      procedure getDensity
      procedure getRefDensity
      procedure getTemperature
      procedure getViscosity
      procedure getThermCond
      procedure getSpecHeat
      procedure getProperty
      procedure getProperty0
      procedure getSatProperty
      procedure getSatTemperature
      procedure checkEnthalpy
      procedure getDensityByMartinelliNelson
   end type

contains

   subroutine reset(this, press)

      class(SteamTable) :: this
      real              :: press

      real              :: steamf
      real              :: enth,  tmodn, dmodn, hmodn                 &
              ,  psia, tm, hdel, tdel, tmodd, dmodd, hmodd

      real              :: prop(NPROP), propn(NPROP), sprop(2*NPROP)

      integer           :: i

      real              :: anistf

      if(abs(this%press-press) .le. 0.1) return

      dmodn = anistf(2250.0, 1, 3, 68.0) ! dummy to initialize nist library.

      this%npnts = nSteamTablePoints
      if(associated(this%hmod)) then
         call this%destroy();
      end if

      allocate(this%hmod(2,0:this%npnts))
      allocate(this%dmodref(2,0:this%npnts))
      allocate(this%propc(NPROP,2,0:this%npnts))
      this%hmod(:,:) = 0.0
      this%dmodref(:,:) = 0.0
      this%propc(:,:,:) = 0.0

      this%press  = press

      ! asme water & steam table (from wce)

      prop(:) = this%getProperty(press, PROP_TEMP, this%tmin)
      this%hmin = prop(PROP_ENTH)
      this%dmin = prop(PROP_DENS)
      this%vismin = prop(PROP_VISC)
      this%tcmin = prop(PROP_TCON)
      this%shmin = prop(PROP_SPCH)

      sprop(:) = this%getSatProperty(press)

      this%tmax   = sprop(PROP_TEMP)
      this%hmax   = sprop(PROP_ENTH)
      this%dmax   = sprop(PROP_DENS)
      this%vismax = sprop(PROP_VISC)
      this%tcmax  = sprop(PROP_TCON)
      this%shmax  = sprop(PROP_SPCH)
      print *, 'MAX : ', this%tmax, this%hmax, this%dmax
      this%hgas   = sprop(PROP_ENTH+NPROP)
      this%dgas   = sprop(PROP_DENS+NPROP)
      this%visgas = sprop(PROP_VISC+NPROP)
      this%tcgas  = sprop(PROP_TCON+NPROP)
      this%shgas  = sprop(PROP_SPCH+NPROP)


      this%rhdiff = 1./(this%hgas - this%hmax)


      hdel = (this%hmax - this%hmin) / this%npnts
      this%rhdel = 1./hdel

      enth = this%hmin

      do i = 1, this%npnts
      enth = enth + hdel

      if(i .eq. this%npnts) enth = enth*0.999  ! to avoid precision problem

      propn(:) = this%getProperty(press, PROP_ENTH, enth)
      tmodn = prop(PROP_TEMP)

      this%propc(:,1,i) = (propn(:)-prop(:))*this%rhdel
      this%propc(:,2,i) = propn(:) - this%propc(:,1,i)*enth

      prop(:) = propn(:)
      enddo

      tdel = (this%tmax - this%tmin) / this%npnts
      this%rtdel = 1./tdel
      print *, this%tmax, this%tmin, this%npnts, tdel
      tm    = this%tmin
      hmodd = this%hmin
      dmodd = this%dmin

      do i = 1, this%npnts
      tm = tm + tdel

      ! to avoid precision problem
      if(i .eq. this%npnts) then
      tm = tm - 0.001
      endif

      prop(:) = this%getProperty(press, PROP_TEMP, tm)

      hmodn = prop(PROP_ENTH)

      this%hmod(1,i) = (hmodn-hmodd)*this%rtdel
      this%hmod(2,i) = hmodn - this%hmod(1,i)*tm

      dmodn = prop(PROP_DENS)

      this%dmodref(1,i) = (dmodn-dmodd)*this%rtdel
      this%dmodref(2,i) = dmodn - this%dmodref(1,i)*tm

      hmodd = hmodn
      dmodd = dmodn
      enddo

      psia = press*bar2psia

      ! modified martinelli-nelson coefficient
      this%cmn(0,1) =  0.5973 - 1.275E-03*psia + 9.019E-07*psia**2 - 2.065E-10*psia**3
      this%cmn(1,1) =  4.7460 + 4.156E-02*psia - 4.011E-05*psia**2 + 9.867E-09*psia**3
      this%cmn(2,1) = -31.270 - 0.5599*psia    + 5.580E-04*psia**2 - 1.378E-07*psia**3
      this%cmn(3,1) =  89.070 + 2.4080*psia    - 2.367E-03*psia**2 + 5.694E-07*psia**3
      this%cmn(0,2) =  0.7847 - 3.900E-04*psia + 1.145E-07*psia**2 - 2.771E-11*psia**3
      this%cmn(1,2) =  0.7707 + 9.619E-04*psia - 2.010E-07*psia**2 + 2.012E-11*psia**3
      this%cmn(2,2) = -1.0600 - 1.194E-03*psia + 2.618E-07*psia**2 - 6.893E-12*psia**3
      this%cmn(3,2) =  0.5157 + 6.506E-04*psia - 1.938E-07*psia**2 + 1.925E-11*psia**3

   end subroutine

   subroutine destroy(this)
      class(SteamTable) :: this

      if(associated(this%propc)) deallocate(this%propc)
      if(associated(this%hmod)) deallocate(this%hmod)
      if(associated(this%dmodref)) deallocate(this%dmodref)

   end subroutine

   function getPressure(this) result(press)
      class(SteamTable) :: this

      real              :: press

      press = this%press

   end function

   function getDensity(this, enthalpy) result(density)
      class(SteamTable) :: this
      real              :: enthalpy, density

      integer           :: index
      real              :: x

      if(enthalpy .le. this%hmax) then
      index = (enthalpy-this%hmin)*this%rhdel + 1
      index = max(index, 1)
      index = min(index, this%npnts)

      density = this%propc(PROP_DENS, 1,index)*enthalpy + this%propc(PROP_DENS, 2,index)
      density = max(density,this%dmax)
      else
         x = (enthalpy-this%hmax) * this%rhdiff

         !modified martinelli-nelson
         density = this%getDensityByMartinelliNelson(x)
      endif

   end function

   function getRefDensity(this, temperature) result(density)
      class(SteamTable) :: this
      real              :: temperature, density

      integer           :: index

      index = (temperature-this%tmin)*this%rtdel + 1
      index = max(index, 1)
      index = min(index, this%npnts)

      density= this%dmodref(1,index)*temperature + this%dmodref(2,index)

      density= max(density,this%dmax)

   end function


   function getDensityByMartinelliNelson(this, x) result(density)

      class(SteamTable) :: this
      real              :: x, density

      real, parameter   :: PRESS_BOUNDARY = 127.55
      real              :: alpha ! void fraction

      if(this%press .lt. PRESS_BOUNDARY) then
      if(x .lt. 0.01) then
         alpha = 0.0
      elseif(x .lt. 0.1) then
         alpha = this%cmn(0,1)+this%cmn(1,1)*x+this%cmn(2,1)*x**2+this%cmn(3,1)*x**3
      elseif(x .lt. 0.9) then
         alpha = this%cmn(0,2)+this%cmn(1,2)*x+this%cmn(2,2)*x**2+this%cmn(3,2)*x**3
      else
         alpha = 1.0
      endif
      else
         alpha = x/this%dgas/((1.0-x)/this%dmax+x/this%dgas)
      endif

      density = (1.-alpha)*this%dmax+alpha*this%dgas
   end function

   function getTemperature(this, enthalpy) result(temperature)
      class(SteamTable) :: this
      real              :: enthalpy, temperature

      integer           :: index

      index = (enthalpy-this%hmin)*this%rhdel + 1
      index = max(index, 1)
      index = min(index, this%npnts)

      temperature = this%propc(PROP_TEMP, 1,index)*enthalpy + this%propc(PROP_TEMP, 2,index)
      temperature = min(temperature,this%tmax)

   end function

   function getViscosity(this, enthalpy) result(viscosity)
      class(SteamTable) :: this
      real              :: enthalpy, viscosity

      integer           :: index

      index = (enthalpy-this%hmin)*this%rhdel + 1
      index = max(index, 1)
      index = min(index, this%npnts)

      viscosity = this%propc(PROP_VISC, 1,index)*enthalpy + this%propc(PROP_VISC, 2,index)
      viscosity = max(viscosity,this%vismax)

   end function

   function getSpecHeat(this, enthalpy) result(specheat)
      class(SteamTable) :: this
      real              :: enthalpy, specheat

      integer           :: index

      index = (enthalpy-this%hmin)*this%rhdel + 1
      index = max(index, 1)
      index = min(index, this%npnts)

      specheat = this%propc(PROP_SPCH, 1,index)*enthalpy + this%propc(PROP_SPCH, 2,index)
      specheat = min(specheat,this%shmax)

   end function

   function getThermCond(this, enthalpy) result(thermcond)
      class(SteamTable) :: this
      real              :: enthalpy, thermcond

      integer           :: index

      index = (enthalpy-this%hmin)*this%rhdel + 1
      index = max(index, 1)
      index = min(index, this%npnts)

      thermcond = this%propc(PROP_TCON, 1,index)*enthalpy + this%propc(PROP_TCON, 2,index)
      thermcond = max(thermcond,this%tcmax)

   end function


   function getEnthalpy(this, temperature) result(enthalpy)
      class(SteamTable) :: this
      real              :: temperature, enthalpy

      integer           :: index

      index = (temperature-this%tmin)*this%rtdel + 1
      print *, 'INDEX : ', temperature, this%tmin, this%rtdel, index
      index = max(index, 1)
      index = min(index, this%npnts)
      enthalpy= this%hmod(1,index)*temperature + this%hmod(2,index)

      enthalpy = min(enthalpy,this%hmax)

   end function

   function getSatTemperature(this) result(temperature)
      class(SteamTable) :: this
      real              :: temperature

      temperature = this%tmax

   end function

   function checkEnthalpy(this, enthalpy) result(ierr)
      class(SteamTable) :: this
      real              :: enthalpy
      integer           :: ierr

      integer           :: index

      if(enthalpy .gt. this%hmax) then
      ierr = STEAM_TABLE_ERROR_MAXENTH
      else
         ierr = STEAM_TABLE_ERROR_NO
      endif
   end function

   ! temp : temeprature
   ! dens : density
   ! property: want property option (1~6)
   function getProperty0(this, temp, dens) result(property)

      integer, parameter  ::  NISTPROP=43, NRIMAX=4

      class(SteamTable)   :: this
      real                ::  temp, dens

      real                ::  property(NPROP)
      integer             ::  iwork(NISTPROP), iwant(NISTPROP), iriflg(nrimax), ipchk(5), ipflg(5)
      real                ::  propr(NISTPROP), propsi(NISTPROP), wavri(NRIMAX), ri(NRIMAX)
      integer             ::  i2ph, ierr, icchk, ischk, i2phck, nri, icflg, igflg, isflg
      real                ::  d1, dv, dl, x, rho, dpd


      iwant =1
      i2phck = 0
      ischk = 0
      icchk = 0
      ipchk = 0
      igflg = 0
      nri = 0

      call props(iwant, temp+KELVIN, dens*1000.0, propsi, propr, i2phck, i2ph,     &
              ischk, isflg, icchk, icflg, ipchk, ipflg, igflg,   &
              nri, wavri, ri, iriflg)

      property(1) = temp                  ! temperature          C
      property(2) = propsi(6)             ! enthalpy             KJ/KG or J/G
      property(3) = dens                  ! density              kg/m^3 -> G/CC
      property(4) = propsi(21) * 10       ! viscosity            Pa-sec -> G/CM-sec
      property(5) = propsi(20) * 0.01     ! thermal conductivity W/m/K -> W/CM/C
      property(6) = propsi(9)             ! specific heat        kJ/kG/K or J/G/C

      return
   end function

   ! igiven  : given property option (1 or 2)
   ! vgiven  : given property value
   ! property: want property option (1~6)
   function getProperty(this, press, igiven, vgiven) result(property)

      integer, parameter  ::  NISTPROP=43, NRIMAX=4

      class(SteamTable)   ::  this
      real                ::  press, vgiven

      integer             ::  igiven
      real                ::  temp, property(NPROP)

      integer             ::  iwork(NISTPROP)
      real                ::  propr(NISTPROP)
      integer             ::  i2ph, ierr
      real                ::  d1, dv, dl, x, rho, dpd



      temp= 0.0

      if(igiven .eq. PROP_ENTH) then
         call hssolv(1, press*bar2mpa, vgiven, temp, d1, dv, dl, i2ph, x, iwork, propr, ierr)
      else
         temp = vgiven + KELVIN
      endif

      iwork = 0
      rho = 0.0
      dpd = 0.0
      propr =0.0
      ierr = 0


      call dens0(rho, press*bar2mpa, temp, dpd, iwork, propr, ierr)

      property = this%getProperty0(temp-KELVIN, rho*RMILLI)

      return
   end function


   ! property: want property option (1~6)
   function getSatProperty(this, press) result(property)

      integer, parameter  ::  NISTPROP=43, NRIMAX=4

      class(SteamTable)   :: this
      real                ::  press
      real                ::  property(NPROP*2)

      integer             ::  iwork(NISTPROP)
      real                ::  propr(NISTPROP)
      integer             ::  ierr
      real                ::  temp, rhof, rhog



      temp = 0.0
      rhof = 0.0
      rhog = 0.0
      iwork(:) = 0
      propr(:) = 0.0
      ierr = 0
      call tsat(press*bar2mpa, temp, rhof, rhog, iwork, propr, ierr)
      print *, 'getSatProperty : ',press*bar2mpa, ierr
      temp = temp - KELVIN
      property(1:NPROP) = this%getProperty0(temp, rhof*RMILLI)
      property(NPROP+1:2*NPROP) = this%getProperty0(temp, rhog*RMILLI)

      return
   end function
end module
   
