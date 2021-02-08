      function anistf(P1, I1, I2, V1)                                          ! Pref  : Pressure                              [psia]
                                                                                ! I1    : given property option (1 or 2)              
!****************************************************************               ! I2    : wanted property option (1 ~ 6)              
!                                                               !               ! V1    : given property                              
!   COPYRIGHT (C) 2008 Korea Nuclear Fuel Co., LTD              !
!   ALL RIGHTS RESERVED                                         !
!                                                               !
!****************************************************************
!     Function anistf is to calculate the properties of water,  !
!     steam and theirs mixture, using NIST water and steam      !
!     Table.                                                    !
!****************************************************************

      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      include 'block.data'
      INCLUDE 'nprop.cmn'

C - ALLOCATE ARRAYS FOR CALCULATIONS, PROPERTIES, AND ALSO BOUNDARY
C - CHECKS PERFORMED UNDER PROPS

      DIMENSION IWORK(NPROP), IWANT(NPROP), PROPR(NPROP), PROPSI(NPROP)
      DIMENSION WAVRI(NRIMAX), RI(NRIMAX), IRIFLG(NRIMAX)
      DIMENSION IPCHK(5), IPFLG(5)

      if(I1.eq.I2) stop ' steam table input error, I1 = I2'

c     Unit conversion British unit -> SI uint

      PP = P1*6.894757D-3                                                       ! pressure             : 1 psia = 6.894757D-3 MPa from ASME

      if(I1 .eq. 1) then

        TT = 5./9.*(V1-32.)+273.15                                              ! temperature          : T(^oK) = 5/9(T(^oF)-32)+273.15

      elseif(I1 .eq. 2) then

        HH = V1*2.326                                                           ! enthalpy             : 1 Btu/lbm = 2.326 KJ/KG from ASME

c     calculation of temperature

        call HSSOLV(1, PP, HH, TT, D1, DV, DL, I2PH, X,
     >              IWORK, PROPR, IERR)

        if(i2 .eq. 1) then
          anistf = 9./5.*(TT-273.15)+32.                                        ! temperature
          return
        endif
        
        if(I2PH .eq. 2) then
          write(*,*) ' Two Phase region'
          write(*,*) ' properties must be calculated in main program '
          stop
        endif
        
      else

        stop ' steam table input error, I1 = 1 or 2'

      endif
      
c     calculation of density
	
      call DENS0(Rho, PP, TT, DPD, IWORK, PROPR, IERR)
	
c     calculation of properties

      IWANT =1
      I2PHCK = 0
      ISCHK = 0
      ICCHK = 0
      IPCHK = 0
      IGFLG = 0
      NRI = 0
      CALL PROPS(IWANT, TT, Rho, PROPSI, PROPR, I2PHCK, I2PH,
     >           ISCHK, ISFLG, ICCHK, ICFLG, IPCHK, IPFLG, IGFLG,
     >           NRI, WAVRI, RI, IRIFLG)

      if(i2.eq.2) anistf = propsi(6)/2.326                                     ! enthalpy
      if(i2.eq.3) anistf = 1./Rho*1.601846D+1                                  ! specific volume      : 1 m^3/kg = 1.601846D+1 ft^3/lbm from ASME
      if(i2.eq.4) anistf = propsi(21)*2.419088D+3                              ! viscosity            : 1 Pa-sec = 2.419088D+3 lbm/ft/hr from ASME
      if(i2.eq.5) anistf = propsi(20)/1.730735                                 ! thermal conductivity : 1 W/m/K = 1/1.730735 Btu/ft/hr/^oF from ASME
      if(i2.eq.6) anistf = propsi(9)/4.1868                                    ! specific heat        : 1 kJ/kG/^oK = 1/4.1868 Btu/lbm/^oF from ASME

      return
      end