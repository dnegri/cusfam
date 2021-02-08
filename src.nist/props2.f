C
      SUBROUTINE PROPS2(IWANT, IGFLG, TR, DEL, PROPR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PROPS2 ***                                   C
C THIS ROUTINE COMPUTES ALL REDUCED THERMODYNAMIC PROPERTIES REQUESTED C
C VIA THE IWANT VECTOR ACCORDING TO THE PRUSS-WAGNER EOS FOR STEAM.    C
C AT A GIVEN REDUCED TEMPERATURE AND DENSITY                           C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C IWANT   IA   I   FOR WHETHER OR NOT TO COMPUTE VARIOUS PROPERTIES,   C
C                  SEE PROPR ARRAY BELOW FOR NUMBERING OF PROPERTIES.  C
C                  1=COMPUTE, 0=DON'T COMPUTE                          C
C IGFLG   I    I   WHETHER TO COMPUTE AND RETURN ONLY IDEAL-GAS PROPS  C
C                  1=IDEAL GAS, 0=REAL FLUID                           C
C TR      R    I   DIMENSIONLESS TEMPERATURE T/TC                      C
C DEL     R    I   DIMENSIONLESS DENSITY RHO/RHOC                      C
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
C 20JUL95 - AHH: ADD FUGACITY COEFFICIENT IN SLOT 19                   C
C 18AUG95 - REVISED VERSION AFTER DEBUGGING BY AHH AND JW              C
C 21AUG95 - AHH: FIX FUGACITY COEFFICIENT COMPUTATION                  C
C 05SEP95 - AHH: USE TR INSTEAD OF TAU IN ARGUMENT LIST                C
C 11SEP95 - AHH: RETURN ZERO FUGACITY IF PRESSURE IS NEGATIVE          C
C 19SEP95 - FOR GUI, MAKE ANY IWANT .NE. 0  GIVE PROPERTY COMPUTATION  C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 07DEC95 - AHH: ADD VOLUME EXPANSIVITY AS PROPERTY 22                 C
C 11DEC95 - AHH: REORDER NUMBERING OF PROPERTY VECTOR                  C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 08JAN97 - AHH: ADD FLAG FOR COMPUTING IDEAL-GAS PROPERTIES           C
C 17JAN97 - AHH: CHANGE ARGUMENT LISTS IN ARESID AND AIDEAL            C
C 11FEB97 - AHH: MAKE EXP'S UNIFORMLY DOUBLE PRECISION                 C
C 27FEB97 - AHH: ADD 8 SLOTS FOR DERIVATIVES (23-30)                   C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWANT(NPROP)
      LOGICAL LPHI, LPHID, LPHIDD, LPHIT, LPHITT, LPHIDT
      LOGICAL LPHDDD, LPHDDT, LPHDTT
C
C INITIALIZE LOGICALS TO FALSE
C
      LPHI = .FALSE.
      LPHID = .FALSE.
      LPHIDD = .FALSE.
      LPHIT = .FALSE.
      LPHITT = .FALSE.
      LPHIDT = .FALSE.
      LPHDDD = .FALSE.
      LPHDDT = .FALSE.
      LPHDTT = .FALSE.
C
      TAU = 1.D0/TR
C
C NOW LOOP THROUGH AND SET WHICH PARTS OF THE HELMHOLTZ ENERGY YOU HAVE
C TO COMPUTE BASED ON THE PROPERTIES REQUESTED
C
      IF (IWANT(2) .NE. 0) THEN
        LPHID = .TRUE.
      ENDIF
      IF (IWANT(6) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIT = .TRUE.
      ENDIF
      IF (IWANT(7) .NE. 0) THEN
        LPHI = .TRUE.
        LPHIT = .TRUE.
      ENDIF
      IF (IWANT(8) .NE. 0) THEN
        LPHITT = .TRUE.
      ENDIF
      IF (IWANT(9) .NE. 0) THEN
        LPHITT = .TRUE.
        LPHID = .TRUE.
        LPHIDD = .TRUE.
        LPHIDT = .TRUE.
      ENDIF
      IF (IWANT(10) .NE. 0) THEN
        LPHIT = .TRUE.
      ENDIF
      IF (IWANT(11) .NE. 0) THEN
        LPHI = .TRUE.
      ENDIF
      IF (IWANT(12) .NE. 0) THEN
        LPHI = .TRUE.
        LPHID = .TRUE.
      ENDIF
      IF (IWANT(13) .NE. 0) THEN
        LPHI = .TRUE.
        LPHID = .TRUE.
      ENDIF
      IF ((IWANT(16) .NE. 0) .OR. (IWANT(14) .NE. 0) .OR.
     >    (IWANT(27) .NE. 0)) THEN
        LPHID = .TRUE.
        LPHIDD = .TRUE.
      ENDIF
      IF (IWANT(15) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIDD = .TRUE.
        LPHIDT = .TRUE.
      ENDIF
      IF (IWANT(17) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIDT = .TRUE.
      ENDIF
      IF (IWANT(18) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIDD = .TRUE.
        LPHITT = .TRUE.
        LPHIDT = .TRUE.
      ENDIF
      IF (IWANT(19) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIDD = .TRUE.
        LPHITT = .TRUE.
        LPHIDT = .TRUE.
      ENDIF
      IF (IWANT(23) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIDD = .TRUE.
        LPHDDD = .TRUE.
      ENDIF
      IF (IWANT(24) .NE. 0) THEN
        LPHDTT = .TRUE.
      ENDIF
      IF (IWANT(25) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIDD = .TRUE.
        LPHIDT = .TRUE.
        LPHDDT = .TRUE.
      ENDIF
      IF (IWANT(26) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIDD = .TRUE.
        LPHIDT = .TRUE.
      ENDIF
      IF (IWANT(28) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIDD = .TRUE.
        LPHIDT = .TRUE.
        LPHDDD = .TRUE.
        LPHDDT = .TRUE.
        LPHDTT = .TRUE.
      ENDIF
      IF (IWANT(29) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIDD = .TRUE.
        LPHDDD = .TRUE.
      ENDIF
      IF (IWANT(30) .NE. 0) THEN
        LPHID = .TRUE.
        LPHIDD = .TRUE.
        LPHIDT = .TRUE.
        LPHDDD = .TRUE.
        LPHDDT = .TRUE.
      ENDIF
C
C NOW CALL THE IDEAL AND (UNLESS IGFLG=1) RESIDUAL HELMHOLTZ ENERGIES
C
      CALL AIDEAL(LPHI, LPHID, LPHIT, LPHIDD, LPHITT, LPHIDT,
     1            LPHDDD, LPHDDT, LPHDTT, PHI0, PHID0, PHIT0, 
     2            PHIDD0, PHITT0, PHIDT0, PHDDD0, PHDDT0, PHDTT0,
     3            TAU, DEL)
      IF (IGFLG .NE. 1) THEN
        CALL ARESID(LPHI, LPHID, LPHIT, LPHIDD, LPHITT, LPHIDT,
     1              LPHDDD, LPHDDT, LPHDTT, PHIR, PHIDR, PHITR, 
     2              PHIDDR, PHITTR, PHIDTR, PHDDDR, PHDDTR, PHDTTR,
     3              TAU, DEL)
      ELSE
        PHIR = 0.D0
        PHIDR = 0.D0
        PHIDDR = 0.D0
        PHITR = 0.D0
        PHITTR = 0.D0
        PHIDTR = 0.D0
        PHDDDR = 0.D0
        PHDDTR = 0.D0
        PHDTTR = 0.D0
      ENDIF
C
C NOW PUT THE TERMS TOGETHER TO CALCULATE THE REQUESTED REDUCED PROPS
C
      IF (IWANT(2) .NE. 0) THEN
        PROPR(2) = 1.D0 + DEL*PHIDR
      ENDIF
      IF (IWANT(6) .NE. 0) THEN
        PROPR(6) = 1.D0 + TAU*(PHIT0+PHITR) + DEL*PHIDR
      ENDIF
      IF (IWANT(7) .NE. 0) THEN
        PROPR(7) = TAU*(PHIT0+PHITR) - PHI0 - PHIR
      ENDIF
      IF (IWANT(8) .NE. 0) THEN
        PROPR(8) = -TAU*TAU*(PHITT0+PHITTR)
      ENDIF
      IF (IWANT(9) .NE. 0) THEN
        PROPR(9) = -TAU*TAU*(PHITT0+PHITTR) +
     1             (1.D0+DEL*PHIDR-DEL*TAU*PHIDTR)**2 /
     2             (1.D0+2.D0*DEL*PHIDR+DEL*DEL*PHIDDR) 
      ENDIF
      IF (IWANT(10) .NE. 0) THEN
        PROPR(10) = TAU*(PHIT0+PHITR)
      ENDIF
      IF (IWANT(11) .NE. 0) THEN
        PROPR(11) = PHI0 + PHIR
      ENDIF
      IF (IWANT(12) .NE. 0) THEN
        PROPR(12) = PHI0 + PHIR + 1.D0 + DEL*PHIDR
      ENDIF
      IF (IWANT(13) .NE. 0) THEN
C ALSO RETURN PRESSURE SO WILL BE ABLE TO COMPUTE FUGACITY
        PROPR(2) = 1.D0 + DEL*PHIDR
        IF (PROPR(2) .GT. 0.D0) THEN
          PROPR(13) = PHIR + (PROPR(2)-1.D0) - LOG(PROPR(2))
          PROPR(13) = DEXP(PROPR(13))
        ELSE
          PROPR(13) = 0.D0
        ENDIF
      ENDIF
      IF (IWANT(14) .NE. 0) THEN
        PROPR(14) = 1.D0 / (1.D0 + 2.D0*DEL*PHIDR + DEL*DEL*PHIDDR)
      ENDIF
      IF (IWANT(15) .NE. 0) THEN
        PROPR(15) = (1.D0+ DEL*PHIDR - TAU*DEL*PHIDTR) /
     >              (1.D0 + 2.D0*DEL*PHIDR + DEL*DEL*PHIDDR)
      ENDIF
      IF (IWANT(16) .NE. 0) THEN
        PROPR(16) = 1.D0 + 2.D0*DEL*PHIDR + DEL*DEL*PHIDDR
      ENDIF
      IF (IWANT(17) .NE. 0) THEN
        PROPR(17) = 1.D0 + DEL*PHIDR - TAU*DEL*PHIDTR
      ENDIF
      IF (IWANT(18) .NE. 0) THEN
        PROPR(18) = 1.D0 + 2.D0*DEL*PHIDR + DEL*DEL*PHIDDR -
     1              (1.D0+DEL*PHIDR-DEL*TAU*PHIDTR)**2 /
     2              (TAU*TAU*(PHITT0+PHITTR))
      ENDIF
      IF (IWANT(19) .NE. 0) THEN
        PROPR(19) = -(DEL*PHIDR+DEL*DEL*PHIDDR+DEL*TAU*PHIDTR) /
     1              ((1.D0+DEL*PHIDR-DEL*TAU*PHIDTR)**2 -
     2               TAU*TAU*(PHITT0+PHITTR)*
     3               (1.D0+2.D0*DEL*PHIDR+DEL*DEL*PHIDDR))
      ENDIF
      IF (IWANT(23) .NE. 0) THEN
        PROPR(23) = DEL * (2.D0*PHIDR + 4.D0*DEL*PHIDDR +
     1                     DEL*DEL*PHDDDR)
      ENDIF
      IF (IWANT(24) .NE. 0) THEN
        PROPR(24) = DEL*TAU*TAU*PHDTTR
      ENDIF
      IF (IWANT(25) .NE. 0) THEN
        PROPR(25) = 1.D0 + 2.D0*DEL*PHIDR - 2.D0*DEL*TAU*PHIDTR +
     1              DEL*DEL * (PHIDDR-TAU*PHDDTR)
      ENDIF 
C
C FOR DERIVATIVES OF DENSITY, COMPUTE SOME QUANTITIES USED IN COMMON
C
      IF ((IWANT(26).NE.0) .OR. (IWANT(27).NE.0) .OR. (IWANT(28).NE.0)
     1    .OR. (IWANT(29).NE.0) .OR. (IWANT(30).NE.0)) THEN
        C = 1.D0 + 2.D0*DEL*PHIDR + DEL*DEL*PHIDDR
      ENDIF
      IF ((IWANT(26).NE.0) .OR. (IWANT(28).NE.0) .OR.
     1    (IWANT(30).NE.0)) THEN
        B = 1.D0 + DEL*PHIDR - DEL*TAU*PHIDTR
        DRT = -B / C
      ENDIF
      IF ((IWANT(28).NE.0) .OR. (IWANT(30).NE.0)) THEN
        DCR = DEL * (2.D0*PHIDR + 4.D0*DEL*PHIDDR + DEL*DEL*PHDDDR)
        DCT = -DEL*TAU * (2.D0*PHIDTR + DEL*PHDDTR)
      ENDIF
C
      IF (IWANT(26) .NE. 0) THEN
        PROPR(26) = DRT
      ENDIF
      IF (IWANT(27) .NE. 0) THEN
        PROPR(27) = 1.D0 / C
      ENDIF
      IF (IWANT(28) .NE. 0) THEN
        DBT = DEL*TAU*TAU*PHDTTR
        DBR = DEL * (PHIDR + DEL*PHIDDR - TAU*PHIDTR - DEL*TAU*PHDDTR)
        DXT = (B - DBT + B*DCT/C) / C
        DXR = -(B + DBR - B*DCR/C) / C
        PROPR(28) = DXT + DXR*DRT
      ENDIF
      IF (IWANT(29) .NE. 0) THEN
        DRP = 1.D0 / C
        D2PDRR = DEL * (2.D0*PHIDR + 4.D0*DEL*PHIDDR +
     1                  DEL*DEL*PHDDDR)
        PROPR(29) = -DRP**3 * D2PDRR
      ENDIF
      IF (IWANT(30) .NE. 0) THEN
        DXT = -(1.D0 + DCT/C) / C
        DXR = -DCR / (C*C)
        PROPR(30) = DXT + DXR*DRT
      ENDIF
      RETURN
      END
