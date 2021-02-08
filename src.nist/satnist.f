      subroutine satnist(P1,TS,Hf,SVf,VISf,TCf,SHf,Hg,SVg,VISg,TCg,SHg)         ! P1    : pressure                              [psia]
                                                                                ! TS    : saturated temperature                 [^oF]
!****************************************************************               ! Hf    : saturated fluid enthalpy              [Btu/lbm]
!                                                               !               ! SVf   : saturated fluid specific volume       [ft^3/lbm]
!   COPYRIGHT (C) 2008 Korea Nuclear Fuel Co., LTD              !               ! VISf  : saturated fluid viscosity             [lbm/ft/hr]
!   ALL RIGHTS RESERVED                                         !               ! TCf   : saturated fluid thermal conductivity  [Btu/ft/hr/^oF]
!                                                               !               ! SHf   : saturated fluid specific heat         [Btu/lbm/^oF]
!****************************************************************               ! XXg   : saturated gas XX
!     Function satnist is to calculate the saturated properties !
!     of water and steam, using NIST water and steam Table.     !
!****************************************************************


      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'

      DIMENSION IWORK(NPROP), IWANT(NPROP), PROPR(NPROP), PROPSI(NPROP)
      DIMENSION WAVRI(NRIMAX), RI(NRIMAX), IRIFLG(NRIMAX)
      DIMENSION IPCHK(5), IPFLG(5)

c     Unit conversion British unit -> SI uint

      PP = P1*6.894757D-3                                                       ! pressure             : 1 psia = 6.894757D-3 MPa from ASME

      call TSAT(PP, TS1, Rhof, Rhog, IWORK, PROPR, IERR)

      TS = 9./5.*(TS1-273.15)+32.
      SVf = 1./Rhof*1.601846D+1                                                 ! specific volume      : 1 m^3/kg = 1.601846D+1 ft^3/lbm from ASME
      SVg = 1./Rhog*1.601846D+1
      
      IWANT = 1
      NRI = 1
      
c     calculation of saturated fluid properties
      
      call PROPS(IWANT, TS1, Rhof, PROPSI, PROPR, 0, I2PH,
     >           ISCHK, ISFLG, ICCHK, ICFLG, ipchk, IPFLG, 0,
     >           NRI, WAVRI, RI, IRIFLG)

      Hf = propsi(6)/2.326                                                      ! enthalpy
      VISf = propsi(21)*2.419088D+3                                             ! viscosity            : 1 Pa-sec = 2.419088D+3 lbm/ft/hr from ASME
      TCf = propsi(20)/1.730735                                                 ! thermal conductivity : 1 W/m/K = 1/1.730735 Btu/ft/hr/^oF from ASME
      SHf = propsi(9)/4.1868                                                    ! specific heat        : 1 kJ/kG/^oK = 1/4.1868 Btu/lbm/^oF from ASME

c     calculation of saturated gas properties

      call PROPS(IWANT, TS1, Rhog, PROPSI, PROPR, 0, I2PH,
     >           ISCHK, ISFLG, ICCHK, ICFLG, ipchk, IPFLG, 0,
     >           NRI, WAVRI, RI, IRIFLG)

      Hg = propsi(6)/2.326                                                      ! enthalpy
      VISg = propsi(21)*2.419088D+3                                             ! viscosity            : 1 Pa-sec = 2.419088D+3 lbm/ft/hr from ASME
      TCg = propsi(20)/1.730735                                                 ! thermal conductivity : 1 W/m/K = 1/1.730735 Btu/ft/hr/^oF from ASME
      SHg = propsi(9)/4.1868                                                    ! specific heat        : 1 kJ/kG/^oK = 1/4.1868 Btu/lbm/^oF from ASME

      return
      end