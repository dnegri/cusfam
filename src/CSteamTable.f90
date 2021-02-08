module CSteamTable
   implicit none
   public
   
   
   real, parameter      :: psia2bar    = 6.894757E-02,  &
                           ft3lbm2gcc  = 1.601846E-02,  &
                           btulbm2kjkg = 2.326,         &
                           bar2psia    = 1./psia2bar,   &
                           kjkg2btulbm = 1./btulbm2kjkg
   
   integer, public, parameter   :: STEAM_TABLE_ERROR_NO      = 0
   integer, public, parameter   :: STEAM_TABLE_ERROR_MAXENTH = 1
   integer, public, parameter   :: STEAM_TABLE_ERROR_MINENTH = 2
   integer, public, parameter   :: STEAM_TABLE_ERROR_MAXTEMP = 3
   integer, public, parameter   :: STEAM_TABLE_ERROR_MINTEMP = 4

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
                     ,  rhdel = 0.0    &
                     ,  rtdel = 0.0    &
                     ,  rhdiff= 0.0    &
                     ,  svg   = 0.0    &
                     ,  svf   = 0.0
      
      real           :: cmn(0:3,2)
      
      real, pointer  :: tmod(:,:)      &
                     ,  dmod(:,:)      &
                     ,  hmod(:,:)      &
                     ,  dmodref(:,:)
      
   contains
      !procedure reset
      !procedure destroy
      !procedure getPressure
      !procedure getDensity
      !procedure getRefDensity
      !procedure getTemperature
      !procedure getEnthalpy
      !procedure getSatTemperature
      !procedure checkEnthalpy
      !procedure getDensityByMartinelliNelson
   end type
   
   contains
   
   
   subroutine reset(this, press)
      
      type(SteamTable), pointer :: this
      real(4)           :: press
      integer           :: npnts
      
      real              :: steamf
      real              :: sf, svg, sg, enth, rbtulbm2kjkg          &
                        ,  psia, tm, hdel, tdel, tmodd, dmodd, hmodd    &
                        ,  tmodn, dmodn, hmodn, tdelc, tmc
      
      integer           :: i

      if(abs(this.press-press) .le. 0.1) return
      this.press = press      
      this.npnts = nSteamTablePoints
      
      if(.not.associated(this.hmod)) then
          allocate(this.hmod(2,0:this.npnts))
          allocate(this.tmod(2,0:this.npnts))
          allocate(this.dmod(2,0:this.npnts))
          allocate(this.dmodref(2,0:this.npnts))
          this.hmod(:,:) = 0.0
          this.tmod(:,:) = 0.0
          this.dmod(:,:) = 0.0
          this.dmodref(:,:) = 0.0
      endif
      
      psia = press*bar2psia
      
      ! asme water & steam table (from wce)   
      sf  = 0.0
      sg  = 0.0
      enth = 0.0
      rbtulbm2kjkg = 1./btulbm2kjkg
      this.hmin = steamf(psia, 1, 2, (this.tmin*1.8+32))
      this.dmin = steamf(psia, 1, 3, (this.tmin*1.8+32))
      this.dmin = 1./this.dmin * ft3lbm2gcc      
      this.hmin = this.hmin*btulbm2kjkg

      call satur(psia,this.tmax,this.svf,this.hmax,sf,this.svg,this.hgas,sg,1)

      this.hgas = this.hgas*btulbm2kjkg
      this.hmax = this.hmax*btulbm2kjkg
      this.rhdiff = 1./(this.hgas - this.hmax)

      this.dgas = 1./this.svg * ft3lbm2gcc
      this.dmax = 1./this.svf * ft3lbm2gcc
      
      this.tmax = (this.tmax-32.)*5./9.
            
      hdel = (this.hmax - this.hmin) / this.npnts
      this.rhdel = 1./hdel
      hdel = hdel * rbtulbm2kjkg
      
      enth = this.hmin * rbtulbm2kjkg
      
      tmodd = this.tmin
      dmodd = this.dmin
      do i = 1, this.npnts
         enth = enth + hdel    ! BTU/LBM
         
         if(i .eq. this.npnts) enth = enth*0.999  ! to avoid precision problem
         
         tmodn = steamf(psia,2,1,enth)
         dmodn = steamf(psia,2,3,enth)

         tmodn = (tmodn-32.)*5./9.
         dmodn = 1./dmodn * ft3lbm2gcc
         
         this.tmod(1,i) = (tmodn-tmodd)*this.rhdel
         this.tmod(2,i) = tmodn - this.tmod(1,i)*enth*btulbm2kjkg
         
         this.dmod(1,i) = (dmodn-dmodd)*this.rhdel
         this.dmod(2,i) = dmodn - this.dmod(1,i)*enth*btulbm2kjkg
         
         tmodd = tmodn
         dmodd = dmodn
      enddo
      
      
      tdel = (this.tmax - this.tmin) / this.npnts
      this.rtdel = 1./tdel
      tdelc=tdel
      tdel = tdel*1.8
      
      tm    = this.tmin*1.8 + 32
      tmc   = this.tmin
      hmodd = this.hmin
      dmodd = this.dmin

      do i = 1, this.npnts
         tm = tm + tdel ! F
         
         tmc = tmc + tdelc
         ! to avoid precision problem
         if(i .eq. this.npnts) then
            tm = tm - 0.0018  
            tmc = tmc - 0.001
         endif
         
         hmodn = steamf(psia,1,2,tm) * btulbm2kjkg
         
         this.hmod(1,i) = (hmodn-hmodd)*this.rtdel
         this.hmod(2,i) = hmodn - this.hmod(1,i)*tmc

         dmodn = steamf(psia, 1, 3, tm)
         dmodn = 1./dmodn * ft3lbm2gcc

         this.dmodref(1,i) = (dmodn-dmodd)*this.rtdel
         this.dmodref(2,i) = dmodn - this.dmodref(1,i)*tmc
         
         hmodd = hmodn
         dmodd = dmodn
      enddo

      ! modified martinelli-nelson coefficient
      this.cmn(0,1) =  0.5973 - 1.275E-03*psia + 9.019E-07*psia**2 - 2.065E-10*psia**3
      this.cmn(1,1) =  4.7460 + 4.156E-02*psia - 4.011E-05*psia**2 + 9.867E-09*psia**3
      this.cmn(2,1) = -31.270 - 0.5599*psia    + 5.580E-04*psia**2 - 1.378E-07*psia**3
      this.cmn(3,1) =  89.070 + 2.4080*psia    - 2.367E-03*psia**2 + 5.694E-07*psia**3
      this.cmn(0,2) =  0.7847 - 3.900E-04*psia + 1.145E-07*psia**2 - 2.771E-11*psia**3
      this.cmn(1,2) =  0.7707 + 9.619E-04*psia - 2.010E-07*psia**2 + 2.012E-11*psia**3
      this.cmn(2,2) = -1.0600 - 1.194E-03*psia + 2.618E-07*psia**2 - 6.893E-12*psia**3
      this.cmn(3,2) =  0.5157 + 6.506E-04*psia - 1.938E-07*psia**2 + 1.925E-11*psia**3
      
   end subroutine
   
   subroutine destroy(this)
      type(SteamTable), pointer :: this
   
      if(associated(this.tmod)) deallocate(this.tmod)
      if(associated(this.dmod)) deallocate(this.dmod)
      if(associated(this.hmod)) deallocate(this.hmod)
      if(associated(this.dmodref)) deallocate(this.dmodref)
      
   end subroutine
      
   function getPressure(this) result(press)
      type(SteamTable), pointer :: this

      real(4)           :: press

      press = this.press

   end function

   function getDensity(this, enthalpy) result(density)
      type(SteamTable), pointer :: this
      real(4)           :: enthalpy, density
      
      integer           :: index
      real(4)           :: x
      
      if(enthalpy .le. this.hmax) then
         index = (enthalpy-this.hmin)*this.rhdel + 1
         index = max(index, 1)
         index = min(index, this.npnts)
      
         density = this.dmod(1,index)*enthalpy + this.dmod(2,index)
         density = max(density,this.dmax)
      else
         x = (enthalpy-this.hmax) * this.rhdiff
         
         !modified martinelli-nelson
         density = getDensityByMartinelliNelson(this, x)
      endif
      
   end function   

   function getRefDensity(this, temperature) result(density)
      type(SteamTable), pointer :: this
      real(4)           :: temperature, density
      
      integer           :: index
      
      index = (temperature-this.tmin)*this.rtdel + 1
      index = max(index, 1)
      index = min(index, this.npnts)
      
      density= this.dmodref(1,index)*temperature + this.dmodref(2,index)
      
      density= max(density,this.dmax)
      
   end function
      
   
   function getDensityByMartinelliNelson(this, x) result(density)
      
      type(SteamTable), pointer :: this
      real(4)           :: x, density
      
      real, parameter   :: PRESS_BOUNDARY = 1850.0*psia2bar
      real              :: alpha ! void fraction
      
      if(this.press .lt. PRESS_BOUNDARY) then
         if(x .lt. 0.01) then
            alpha = 0.0
         elseif(x .lt. 0.1) then
            alpha = this.cmn(0,1)+this.cmn(1,1)*x+this.cmn(2,1)*x**2+this.cmn(3,1)*x**3
         elseif(x .lt. 0.9) then
            alpha = this.cmn(0,2)+this.cmn(1,2)*x+this.cmn(2,2)*x**2+this.cmn(3,2)*x**3
         else
            alpha = 1.0
         endif
      else
         alpha = x*this.svg/((1.0-x)*this.svf+x*this.svg)
      endif
      
      density = (1.-alpha)*this.dmax+alpha*this.dgas
   end function
      
   function getTemperature(this, enthalpy) result(temperature)
      type(SteamTable), pointer :: this
      real(4)           :: enthalpy, temperature
      
      integer           :: index
      
      index = (enthalpy-this.hmin)*this.rhdel + 1
      index = max(index, 1)
      index = min(index, this.npnts)
      
      temperature = this.tmod(1,index)*enthalpy + this.tmod(2,index)
      temperature = min(temperature,this.tmax)

   end function

   function getEnthalpy(this, temperature) result(enthalpy)
      type(SteamTable), pointer :: this
      real(4)           :: temperature, enthalpy
      
      integer           :: index
      
      index = (temperature-this.tmin)*this.rtdel + 1
      index = max(index, 1)
      index = min(index, this.npnts)
      
      enthalpy= this.hmod(1,index)*temperature + this.hmod(2,index)
      
      enthalpy = min(enthalpy,this.hmax)
      
   end function
      
   function getSatTemperature(this) result(temperature)
      type(SteamTable), pointer :: this
      real(4)           :: temperature
      
      temperature = this.tmax

   end function
   
   function checkEnthalpy(this, enthalpy) result(ierr)
      type(SteamTable), pointer :: this
      real(4)           :: enthalpy
      integer           :: ierr
      
      integer           :: index
      
      if(enthalpy .gt. this.hmax) then
         ierr = STEAM_TABLE_ERROR_MAXENTH
      else
         ierr = STEAM_TABLE_ERROR_NO
      endif
   end function   
end module
   
