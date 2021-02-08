C
      SUBROUTINE PROPS1(IWANT, IGFLG, TK, RHO, PROPSI, PROPR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PROPS1 ***                                   C
C THIS ROUTINE COMPUTES ALL PROPERTIES REQUESTED VIA THE IWANT VECTOR  C
C AT A GIVEN TEMPERATURE AND DENSITY.  PROPERTIES ARE RETURNED IN SI   C
C UNITS: KG, M3, K, MPA, KJ.  THE LOWER-LEVEL ROUTINE PROPS2 IS        C
C CALLED TO RETURN THE PROPERTIES IN DIMENSIONLESS FORM. WTRANS IS     C
C CALLED TO RETURN THE VISCOSITY AND THERMAL CONDUCTIVITY. DIELEC IS   C
C CALLED TO RETURN THE DIELECTRIC CONSTANT AND ITS DERIVATIVES         C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C IWANT   IA   I   FOR WHETHER OR NOT TO COMPUTE VARIOUS PROPERTIES,   C
C                  SEE PROPR ARRAY BELOW FOR NUMBERING OF PROPERTIES.  C
C                  1=COMPUTE, 0=DON'T COMPUTE                          C
C IGFLG   I    I   WHETHER TO COMPUTE AND RETURN ONLY IDEAL-GAS PROPS  C
C                  1=IDEAL GAS, 0=REAL FLUID                           C
C TK      R    I   TEMPERATURE IN KELVINS                              C
C RHO     R    I   DENSITY IN KG/M3                                    C
C PROPSI  RA   O   VECTOR OF PROPERTIES IN SI UNITS AS REQUESTED BY    C
C                  IWANT.  NUMBERING OF PROPERTIES:                    C
C                  1: TEMPERATURE IN K                                 C
C                  2: PRESSURE IN MPA                                  C
C                  3: DENSITY IN KG/M3                                 C
C                  4: VOLUME IN M3/KG                                  C
C                  5: QUALITY (ONLY SET IF 2-PHASE)                    C
C                  6: ENTHALPY IN KJ/KG                                C
C                  7: ENTROPY IN KJ/(KG*K)                             C
C                  8: ISOCHORIC HEAT CAPACITY IN KJ/(KG*K)             C
C                  9: ISOBARIC HEAT CAPACITY IN KJ/(KG*K)              C
C                 10: INTERNAL ENERGY IN KJ/KG                         C
C                 11: HELMHOLTZ ENERGY IN KJ/KG                        C
C                 12: GIBBS ENERGY IN KJ/KG                            C
C                 13: FUGACITY IN MPA                                  C
C                 14: ISOTHERMAL COMPRESSIBILITY IN MPA**(-1)          C
C                 15: VOLUME EXPANSIVITY IN K**(-1)                    C
C                 16: DP/DRHO (CONSTANT T) IN MPA/(KG/M3)              C
C                 17: DP/DT (CONSTANT RHO) IN MPA/K                    C
C                 18: SPEED OF SOUND IN M/SEC                          C
C                 19: JOULE-THOMSON COEFFICIENT IN K/MPA               C
C                 20: THERMAL CONDUCTIVITY IN W/M-K                    C
C                 21: VISCOSITY IN PA-SEC                              C
C                 22: STATIC DIELECTRIC CONSTANT (RETURNED AS -999. IF C
C                     CALCULATION COULD NOT BE DONE)                   C
C                 23: 2ND DERIVATIVE D2P/DRHO2 (CONSTANT T) IN         C
C                     MPA/(KG/M3)**2                                   C
C                 24: 2ND DERIVATIVE D2P/DT2 (CONSTANT RHO)            C
C                     IN MPA/K**2                                      C
C                 25: 2ND DERIVATIVE D2P/DRHODT (CONSTANT T, RHO)      C
C                     IN MPA/(KG/M3)/K                                 C
C                 26: DRHO/DT (CONSTANT P) IN (KG/M3)/K                C
C                 27: DRHO/DP (CONSTANT T) IN (KG/M3)/MPA              C
C                 28: 2ND DERIVATIVE D2RHO/DT2 (CONSTANT P)            C
C                     IN (KG/M3)/K**2                                  C
C                 29: 2ND DERIVATIVE D2RHO/DP2 (CONSTANT T)            C
C                     IN (KG/M3)/MPA**2                                C
C                 30: 2ND DERIVATIVE D2RHO/DTDP (CONSTANT P, T)        C
C                     IN (KG/M3)/K/MPA                                 C
C                 31: DIELECTRIC DERIVATIVE D(EPS)/D(RHO) (CONSTANT T) C
C                     IN M3/KG                                         C
C                 32: DIELECTRIC DERIVATIVE D(EPS)/DT (CONSTANT RHO)   C
C                     IN K**(-1)                                       C
C                 33: DIELECTRIC DERIVATIVE D(EPS)/DT (CONSTANT P)     C
C                     IN K**(-1)                                       C
C                 34: DIELECTRIC DERIVATIVE D(EPS)/DP (CONSTANT T)     C
C                     IN MPA**(-1)                                     C
C                 35: 2ND DIELECTRIC DERIVATIVE D2(EPS)/DT2 (CONST. P) C
C                     IN K**(-2)                                       C
C                 36: 2ND DIELECTRIC DERIVATIVE D2(EPS)/DP2 (CONST. T) C
C                     IN MPA**(-2)                                     C
C                 37: 2ND DIELECTRIC DERIVATIVE D2(EPS)/DTDP           C
C                     (CONSTANT P, T) IN (K*MPA)**(-1)                 C
C                 38: DEBYE-HUCKEL SLOPE FOR ACTIVITY COEFFICIENT      C
C                     IN (KG/MOL)**0.5                                 C
C                 39: D-H SLOPE FOR OSMOTIC COEFF. IN (KG/MOL)**0.5    C
C                 40: D-H SLOPE FOR APPARENT MOLAR VOLUME              C
C                     IN (M3/MOL)*(KG/MOL)**0.5                        C
C                 41: D-H SLOPE FOR APPARENT MOLAR ENTHALPY, DIVIDED   C
C                     BY RT, IN (KG/MOL)**0.5                          C
C                 42: D-H SLOPE FOR APPARENT MOLAR COMPRESSIBILITY     C
C                     IN (M3/MOL/MPA)*(KG/MOL)**0.5                    C
C                 43: D-H SLOPE FOR APPARENT MOLAR HEAT CAPACITY,      C
C                     DIVIDED BY R, IN (KG/MOL)**0.5                   C
C PROPR   RA   O   VECTOR OF REDUCED PROPERTIES AS REQUESTED BY IWANT  C
C                  TO BE RETURNED BY THE PROPS2 ROUTINE                C
C                  1: TEMPERATURE (SLOT NOT USED)                      C
C                  2: REDUCED PRESSURE P/(RHO*R*T)                     C
C                  3: DENSITY (SLOT NOT USED)                          C
C                  4: VOLUME (SLOT NOT USED)                           C
C                  5: QUALITY (SLOT NOT USED)                          C
C                  6: REDUCED ENTHALPY H/RT                            C
C                  7: REDUCED ENTROPY S/R                              C
C                  8: REDUCED ISOCHORIC HEAT CAPACITY CV/R             C
C                  9: REDUCED ISOBARIC HEAT CAPACITY CP/R              C
C                 10: REDUCED INTERNAL ENERGY U/RT                     C
C                 11: REDUCED HELMHOLTZ ENERGY A/RT                    C
C                 12: REDUCED GIBBS ENERGY G/RT                        C
C                 13: FUGACITY COEFFICIENT FUG/PRES                    C
C                 14: REDUCED ISOTHERMAL COMPRESSIBILITY (COMPR*RHO*RT,C
C                     RT/DPDR)                                         C
C                 15: REDUCED VOLUME EXPANSIVITY EXPANS*T              C
C                 16: REDUCED DP/DRHO (CONSTANT T) DPDR/RT             C
C                 17: REDUCED DP/DT (CONSTANT RHO) DPDT/(RHO*R)        C
C                 18: REDUCED SPEED OF SOUND (W**2)/RT                 C
C                 19: REDUCED JOULE-THOMSON COEFFICIENT MU*R*RHO       C
C                 20: THERMAL CONDUCTIVITY (SLOT NOT USED)             C
C                 21: VISCOSITY (SLOT NOT USED)                        C
C                 22: DIELECTRIC CONSTANT (SLOT NOT USED)              C
C                 23: REDUCED 2ND DERIVATIVE D2P/DRHO2 (CONSTANT T)    C
C                     D2PDRR*RHO/RT                                    C
C                 24: REDUCED 2ND DERIVATIVE D2P/DT2 (CONSTANT RHO)    C
C                     D2PDTT*T/(RHO*R)                                 C
C                 25: REDUCED 2ND DERIVATIVE D2P/DRHODT (CONSTANT T,   C
C                     RHO) D2PDRT/R                                    C
C                 26: REDUCED DRHO/DT (CONSTANT P) DRDT*T/RHO          C
C                 27: REDUCED DRHO/DP (CONSTANT T) DRDP*RT             C
C                 28: REDUCED 2ND DERIVATIVE D2RHO/DT2 (CONSTANT P)    C
C                     D2RDTT*T*T/RHO                                   C
C                 29: REDUCED 2ND DERIVATIVE D2RHO/DP2 (CONSTANT T)    C
C                     D2RDPP*RHO*(RT**2)                               C
C                 30: REDUCED 2ND DERIVATIVE D2RHO/DTDP (CONSTANT P,T) C
C                     D2RDTP*R*T*T                                     C
C                 31-37: DIELECTRIC DERIVATIVES (SLOTS NOT USED)       C
C                 38-43: DEBYE-HUCKEL SLOPES (SLOTS NOT USED)          C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 11JUL95 - INITIAL CREATION BY AHH                                    C
C 20JUL95 - AHH: ADDITION OF FUGACITY AS PROPERTY 19                   C
C 18AUG95 - VERSION AFTER DEBUGGING BY AHH AND JW                      C
C 05SEP95 - CHANGE TO TR IN PROPS2 ARGUMENT LIST                       C
C 11SEP95 - FILL IN TEMPERATURE AND DENSITY SLOTS IF REQUESTED         C
C 19SEP95 - FOR GUI, MAKE ANY IWANT .NE. 0  GIVE PROPERTY COMPUTATION  C
C 25SEP95 - AHH: PUT IN THERMAL CONDUCTIVITY AND VISCOSITY             C
C 28SEP95 - AHH: RESET IWANT FLAGS CHANGED INTERNALLY                  C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 27NOV95 - AHH: ADD DIELECTRIC CONSTANT IN SLOT 21                    C
C 07DEC95 - AHH: ADD VOLUME EXPANSIVITY AS PROPERTY 22                 C
C 11DEC95 - AHH: REORDER NUMBERING IN PROPERTY VECTORS                 C
C 12DEC95 - AHH: CORRECT EXPANSIVITY CALCULATION                       C
C 15DEC95 - AHH: ALLOW FOR FAILURE OF DIELECTRIC CALCULATION           C
C 15FEB96 - AHH: CHANGE ARGUMENT LIST FOR DIELEC SUBROUTINE            C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 17SEP96 - AHH: EXPUNGE TAB CHARACTERS                                C
C 08JAN97 - AHH: ADD FLAG FOR COMPUTING IDEAL-GAS PROPERTIES           C
C 09JAN97 - AHH: FILL IN OTHER PROPS (DIELEC, TRANS) FOR IDEAL GAS     C
C 09JAN97 - AHH: CHANGE ARGUMENT LIST FOR DIELEC                       C
C 11FEB97 - AHH: MAKE SQRT UNIFORMLY DOUBLE PRECISION                  C
C 26FEB97 - AHH: ADD DERIVATIVE PROPERTIES IN SLOTS 23-30              C
C 05MAR97 - AHH: ADD DIELECTRIC DERIVATIVES IN SLOTS 31-37             C
C 06MAR97 - AHH: ADD DEBYE-HUCKEL SLOPES IN SLOTS 38-43                C
C 18MAY98 - AHH: HANDLE SLOTS 31-43 IF OUT OF RANGE FOR DIELEC CALC    C
C 09AUG99 - AHH: UPDATE TO 1998 FUNDAMENTAL CONSTANTS FOR D-H SLOPES   C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
C
C NEED CONSTANTS FOR WATER 
C
      INCLUDE 'wconst.cmn'
      DIMENSION PROPSI(NPROP), PROPR(NPROP)
      DIMENSION IWANT(NPROP)
      PARAMETER (ECH=1.602176462D-19, BOLTZ=1.3806503D-23,
     >           AV=6.02214199D23)
C
C COMPUTE REDUCED TEMPERATURE AND REDUCED DENSITY
C
      TR  = TK/TCW
      DEL = RHO/RHOCW
      RTW = RW*TK
C
C SAVE SOME IWANT FLAGS THAT MIGHT GET CHANGED WITHIN THE ROUTINE
C
      ISAV14 = IWANT(14)
      ISAV17 = IWANT(17)
      ISAV22 = IWANT(22)
      ISAV23 = IWANT(23)
      ISAV24 = IWANT(24)
      ISAV25 = IWANT(25)
      ISAV26 = IWANT(26)
      ISAV27 = IWANT(27)
      ISAV28 = IWANT(28)
      ISAV29 = IWANT(29)
      ISAV31 = IWANT(31)
      ISAV33 = IWANT(33)
      ISAV34 = IWANT(34)
      ISAV35 = IWANT(35)
      ISAV36 = IWANT(36)
      ISAV37 = IWANT(37)
      ISAV39 = IWANT(39)
      ISAV40 = IWANT(40)
      ISAV41 = IWANT(41)
C
C IF THERMAL CONDUCTIVITY OR VISCOSITY REQUESTED, COMPUTE OTHER 
C PROPERTIES NEEDED FOR THOSE COMPUTATIONS
C
      MODE = 0
      IF (IWANT(20) .GT. 0) THEN
        IWANT(17) = 1
        IWANT(14) = 1
        MODE = MODE + 1
      ENDIF
      IF (IWANT(21) .GT. 0) THEN
        IWANT(14) = 1
        MODE = MODE + 2
      ENDIF
C
C FOR SOME DEBYE-HUCKEL SLOPES, NEED TO COMPUTE CERTAIN DERIVATIVES
C
      MODEDH = 0
      IF (IWANT(43) .NE. 0) THEN
        IWANT(41) = 1
        IWANT(35) = 1
        IWANT(28) = 1
      ENDIF
      IF (IWANT(42) .NE. 0) THEN
        IWANT(40) = 1
        IWANT(36) = 1
        IWANT(29) = 1
      ENDIF
      IF (IWANT(41) .NE. 0) THEN
        IWANT(39) = 1
        IWANT(33) = 1
        IWANT(26) = 1
      ENDIF
      IF (IWANT(40) .NE. 0) THEN
        IWANT(39) = 1
        IWANT(34) = 1
        IWANT(27) = 1
      ENDIF
      IF ((IWANT(39).NE.0) .OR. (IWANT(38).NE.0)) THEN
        IWANT(22) = 1
        MODEDH = 1
      ENDIF
C
C FOR SOME DIELECTRIC DERIVATIVES, ALSO NEED SOME OTHER PROPERTIES
C
      MODED = 0
      IF (IWANT(35) .GT. 0) THEN
        IWANT(24) = 1
        IWANT(37) = 1
        MODED = 2
      ENDIF
      IF (IWANT(36) .GT. 0) THEN
        IWANT(23) = 1
        IWANT(27) = 1
        IWANT(31) = 1
        MODED = 2
      ENDIF
      IF (IWANT(37) .GT. 0) THEN
        IWANT(17) = 1
        IWANT(23) = 1
        IWANT(25) = 1
        IWANT(31) = 1
        IWANT(34) = 1
        MODED = 2
      ENDIF
      IF ((IWANT(31).NE.0) .OR. (IWANT(32).NE.0) .OR.
     1    (IWANT(33).NE.0) .OR. (IWANT(34).NE.0)) THEN
        MODED = MAX0(MODED, 1)
      ENDIF
      IF (IWANT(33) .GT. 0) THEN
        IWANT(17) = 1
        IWANT(27) = 1
      ENDIF
      IF (IWANT(34) .GT. 0) THEN
        IWANT(27) = 1
      ENDIF
C
C CALL PROPS2 TO RETURN DIMENSIONLESS PROPERTIES
C
      CALL PROPS2(IWANT, IGFLG, TR, DEL, PROPR)
C
C NOW FIGURE THE REQUESTED PROPERTIES IN SI UNITS
C
      IF (IWANT(1) .NE. 0) THEN
        PROPSI(1) = TK
      ENDIF
      IF (IWANT(2) .NE. 0) THEN
        PROPSI(2) = PROPR(2)*RTW*RHO*1.D-3
      ENDIF
      IF (IWANT(3) .NE. 0) THEN
        PROPSI(3) = RHO
      ENDIF
      IF (IWANT(4) .NE. 0) THEN
        PROPSI(4) = 1./RHO
      ENDIF
      IF (IWANT(6) .NE. 0) THEN
        PROPSI(6) = PROPR(6)*RTW
      ENDIF
      IF (IWANT(7) .NE. 0) THEN
        PROPSI(7) = PROPR(7)*RW
      ENDIF
      IF (IWANT(8) .NE. 0) THEN
        PROPSI(8) = PROPR(8)*RW
      ENDIF
      IF (IWANT(9) .NE. 0) THEN
        PROPSI(9) = PROPR(9)*RW
      ENDIF
      IF (IWANT(10) .NE. 0) THEN
        PROPSI(10) = PROPR(10)*RTW
      ENDIF
      IF (IWANT(11) .NE. 0) THEN
        PROPSI(11) = PROPR(11)*RTW
      ENDIF
      IF (IWANT(12) .NE. 0) THEN
        PROPSI(12) = PROPR(12)*RTW
      ENDIF
      IF (IWANT(13) .NE. 0) THEN
        PROPSI(2) = PROPR(2)*RTW*RHO*1.D-3
        PROPSI(13) = PROPR(13)*PROPSI(2)
      ENDIF
      IF (IWANT(14) .NE. 0) THEN
        PROPSI(14) = PROPR(14)/RHO/RTW*1.D3
      ENDIF
      IF (IWANT(15) .NE. 0) THEN
        PROPSI(15) = PROPR(15)/TK
      ENDIF
      IF (IWANT(16) .NE. 0) THEN
        PROPSI(16) = PROPR(16)*RTW*1.D-3
      ENDIF
      IF (IWANT(17) .NE. 0) THEN
        PROPSI(17) = PROPR(17)*RHO*RW*1.D-3
      ENDIF
      IF (IWANT(18) .NE. 0) THEN
        IF (PROPR(18) .GT. 0.D0) THEN
          PROPSI(18) = DSQRT(PROPR(18)*RTW*1.D3)
        ELSE
          PROPSI(18) = 0.0D0
        ENDIF
      ENDIF
      IF (IWANT(19) .NE. 0) THEN
        PROPSI(19) = PROPR(19)/RHO/RW*1.D3
      ENDIF        
      IF (IWANT(23) .NE. 0) THEN
        PROPSI(23) = PROPR(23)*RTW/RHO * 1.D-3
      ENDIF
      IF (IWANT(24) .NE. 0) THEN
        PROPSI(24) = PROPR(24)*RHO*RW/TK * 1.D-3
      ENDIF
      IF (IWANT(25) .NE. 0) THEN
        PROPSI(25) = PROPR(25)*RW * 1.D-3
      ENDIF
      IF (IWANT(26) .NE. 0) THEN
        PROPSI(26) = PROPR(26)*RHO/TK
      ENDIF
      IF (IWANT(27) .NE. 0) THEN
        PROPSI(27) = PROPR(27)/RTW * 1.D3
      ENDIF
      IF (IWANT(28) .NE. 0) THEN
        PROPSI(28) = PROPR(28)*RHO/(TK*TK)
      ENDIF
      IF (IWANT(29) .NE. 0) THEN
        PROPSI(29) = PROPR(29)/(RTW*RTW*RHO) * 1.D6
      ENDIF
      IF (IWANT(30) .NE. 0) THEN
        PROPSI(30) = PROPR(30)/(RTW*TK) * 1.D3
      ENDIF
C
      IF ((IWANT(22) .NE. 0) .OR. (MODED .GT. 0)) THEN
        IF (IGFLG .EQ. 1) THEN
          PROPSI(22) = -999.D0
          IF (MODED .GT. 0) THEN
            DO 210 I=31,37
              PROPSI(I) = 0.D0
  210       CONTINUE
          ENDIF
          IF (MODEDH .GT. 0) THEN
            DO 212 I=38,43
              PROPSI(I) = 0.D0
  212       CONTINUE
          ENDIF
        ELSE
C
C FIGURE OUT WHICH DIELECTRIC DERIVATIVES ARE NEEDED
C
          CALL DIELEC(TK, RHO, PROPSI(22), PROPSI(32), PROPSI(31),
     1                D2EDTT, D2EDRR, D2EDTR, MODED, IERD)
          IF (IERD .EQ. 1) THEN
            PROPSI(22) = -999.D0
            IF (MODED .GT. 0) THEN
              DO 220 I=31,37
                PROPSI(I) = 0.D0
  220         CONTINUE
            ENDIF
            IF (MODEDH .GT. 0) THEN
              DO 222 I=38,43
                PROPSI(I) = 0.D0
  222         CONTINUE
            ENDIF
          ELSE
            IF (IWANT(33) .NE. 0) THEN
              PROPSI(33) = PROPSI(32) - PROPSI(31)*PROPSI(17)*PROPSI(27)
            ENDIF
            IF (IWANT(34) .NE. 0) THEN
              PROPSI(34) = PROPSI(31) * PROPSI(27)
            ENDIF
            IF ((IWANT(36) .NE. 0) .OR. (IWANT(37) .NE. 0)) THEN
              DEPTR = PROPSI(27) * (D2EDRR - PROPSI(31)*PROPSI(27)*
     1                              PROPSI(23))
              IF (IWANT(36) .NE. 0) THEN
                PROPSI(36) = PROPSI(27) * DEPTR
              ENDIF
              IF (IWANT(37) .NE. 0) THEN
                PROPSI(37) = PROPSI(27) * (D2EDTR - DEPTR*PROPSI(17) -
     1                                     PROPSI(34)*PROPSI(25))
              ENDIF
            ENDIF
            IF (IWANT(35) .NE. 0) THEN
              THING1 = PROPSI(27) * (D2EDTR - PROPSI(31)*PROPSI(27)*
     1                                        PROPSI(25))
              THING2 = D2EDTT - THING1*PROPSI(17) -
     1                 PROPSI(34)*PROPSI(24)
              PROPSI(35) = THING2 - PROPSI(37)*PROPSI(17)
            ENDIF
          ENDIF
        ENDIF
      ENDIF
C
C FIGURE DEBYE-HUCKEL SLOPES
C
      IF ((MODEDH .EQ. 1) .AND. (IERD .NE. 1)) THEN
        PI = DACOS(-1.D0)
        EPS0 = 1.D0 / (4.D-7*PI*(299792458.D0)**2)
        RGAS = AV*BOLTZ
C
C - NOTE: WILL WRITE A SUB GAMMA AS C1*RHO**0.5 * C2*(EPS*T)**-1.5
C -       THEN A SUB PHI IS (1./3.)*C1*C2*(RHO**0.5)*(EPS*T)**1.5
C
        C1 = 2.D0*PI*AV
        C2 = ECH*ECH / (4.D0*PI*EPS0*BOLTZ)
        C3 = C2*DSQRT(C1*C2)
        EPS = PROPSI(22)
        IF (IWANT(38) .NE. 0) THEN
          PROPSI(38) = C3*DSQRT(RHO)*(EPS*TK)**(-1.5D0)
        ENDIF
        IF (IWANT(39) .NE. 0) THEN
          PROPSI(39) = (C3*DSQRT(RHO)*(EPS*TK)**(-1.5D0)) / 3.D0
        ENDIF
        IF (IWANT(40) .NE. 0) THEN
          PROPSI(40) = 2.D0*RGAS*TK*1.D-6*PROPSI(39) * 
     1                (3.D0*PROPSI(34)/EPS - PROPSI(27)/RHO)
        ENDIF
        IF (IWANT(41) .NE. 0) THEN
          PROPSI(41) = -6.D0*PROPSI(39) * (1.D0 +
     1                 TK*PROPSI(33)/EPS - TK*PROPSI(26)/RHO/3.D0)
        ENDIF
        IF (IWANT(42) .NE. 0) THEN
          PROPSI(42) = -0.5*PROPSI(40) * 
     1                 (3.D0*PROPSI(34)/EPS - PROPSI(27)/RHO) +
     2                 2.D0*RGAS*TK*1.D-6*PROPSI(39) *
     3                 (-3.D0*PROPSI(34)**2/(EPS**2)+3.D0*PROPSI(36)/EPS
     4                  +(PROPSI(27)/RHO)**2-PROPSI(29)/RHO)
        ENDIF
        IF (IWANT(43) .NE. 0) THEN
          THING1 = -1.5D0*PROPSI(41) * (1.D0 +
     1              TK*PROPSI(33)/EPS - TK*PROPSI(26)/RHO/3.D0)
          THING2 = -6.D0*TK*PROPSI(39) * (1.D0/TK+2.D0*PROPSI(33)/EPS
     1              -TK*(PROPSI(33)/EPS)**2+TK*PROPSI(35)/EPS
     2              +(-2.D0*PROPSI(26)/RHO+TK*(PROPSI(26)/RHO)**2
     3                -TK*PROPSI(28)/RHO)/3.D0)
          PROPSI(43) = THING1 + THING2
        ENDIF
      ENDIF
C
C NOW CALL FOR TRANSPORT PROPERTIES IF REQUESTED
C NOTE THAT THE IDEAL-GAS VALUES FOR THESE ARE NOT MEANINGFUL,
C SO SET TO -999. IN THAT CASE.
C
      IF (MODE .GT. 0) THEN
        IF (IGFLG .EQ. 1) THEN
          PROPSI(20) = -999.D0
          PROPSI(21) = -999.D0
        ELSE
          CALL WTRANS(MODE, TK, RHO, PROPSI(14), PROPSI(17),
     >                PROPSI(20), PROPSI(21))
        ENDIF
      ENDIF
C
C RESET ALL SAVED FLAGS
C
      IWANT(14) = ISAV14
      IWANT(17) = ISAV17
      IWANT(22) = ISAV22
      IWANT(23) = ISAV23
      IWANT(24) = ISAV24
      IWANT(25) = ISAV25
      IWANT(26) = ISAV26
      IWANT(27) = ISAV27
      IWANT(28) = ISAV28
      IWANT(29) = ISAV29
      IWANT(31) = ISAV31
      IWANT(33) = ISAV33
      IWANT(34) = ISAV34
      IWANT(35) = ISAV35
      IWANT(36) = ISAV36
      IWANT(37) = ISAV37
      IWANT(39) = ISAV39
      IWANT(40) = ISAV40
      IWANT(41) = ISAV41
C
      RETURN
      END
