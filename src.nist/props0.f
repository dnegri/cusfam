C
      SUBROUTINE PROPS0(IWANT, TK, RHO, PROPSI, PROPR, I2PHCK, I2PH,
     >                  ISCHK, ISFLG, ICCHK, ICFLG, IPCOLD, IPFOLD)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PROPS0 ***                                   C
C THIS ROUTINE COMPUTES ALL PROPERTIES REQUESTED VIA THE IWANT VECTOR  C
C AT A GIVEN TEMPERATURE AND DENSITY.  PROPERTIES ARE RETURNED IN SI   C
C UNITS: KG, M3, K, MPA, KJ.                                           C
C THIS ROUTINE IS OBSOLETE AS OF VERSION 2.1 (PROPS SHOULD BE USED     C
C INSTEAD) BUT IS INCLUDED FOR BACKWARD COMPATIBILITY.                 C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C IWANT   IA   I   FOR WHETHER OR NOT TO COMPUTE VARIOUS PROPERTIES,   C
C                  SEE PROPSI ARRAY BELOW FOR NUMBERING OF PROPERTIES. C
C                  1=COMPUTE, 0=DON'T COMPUTE                          C
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
C PROPR   RA   -  WORK VECTOR FOR REDUCED PROPERTIES PASSED TO PROPS1  C
C I2PHCK  I    I  FLAG FOR WHETHER TO CHECK AND SEE IF REQUESTED T AND C
C                 RHO IN 2-PHASE REGION.  0=DON'T CHECK, 1=CHECK       C
C I2PH    I    O  OUTPUT FLAG FOR RELATION TO 2-PHASE REGION           C
C                 0 = I2PHCK=0 (OR IDEAL GAS)                          C
C                -1 = SUBCOOLED LIQUID (ALSO T<TC AND P>PC)            C                                C
C                 1 = SUPERHEATED VAPOR (ALSO T>TC)                    C
C                 2 = 2-PHASE REGION.  QUALITY IS PUT IN SLOT 5, PROPS C
C                     RETURNED IN PROPSI ARE APPROPRIATE AVERAGES. FOR C
C                     PROPS THAT CAN'T BE AVERAGED, ZERO IS RETURNED   C
C                 3 = UNABLE TO PERFORM 2-PHASE CHECK BECAUSE TK BELOW C
C                     MINIMUM.  PROPS RETURNED AS THOUGH 1-PHASE       C
C                 4 = SAME AS 2, EXCEPT 2-PHASE ENVELOPE EXTRAPOLATED  C
C                     TO REGION BELOW TRIPLE POINT                     C
C *** NOTE:  FOLLOWING ARE FLAGS FOR WHETHER TO MAKE A CHECK (1=CHECK, C
C            0=DON'T) AND FLAGS RETURNED FROM THE CHECKS (0=OK OR NOT  C
C            CHECKED, 1=EXTRAPOLATED BEYOND RECOMMENDED REGION, 2=     C
C            UNACCEPTABLY OUTSIDE OF REGION                            C
C ISCHK   I    I   WHETHER TO CHECK IF EQUILIBRIUM PHASE IS SOLID      C
C ISFLG   I    O   RESULT OF SOLID CHECK                               C
C ICCHK   I    I   WHETHER TO CHECK IF TOO NEAR CRITICAL POINT         C
C ICFLG   I    O   RESULT OF NEAR-CRITICAL CHECK                       C
C IPCHK   IA   I   WHETHER TO CHECK BOUNDARIES FOR VARIOUS PROPERTIES. C
C                  ARRAY ELEMENTS ARE: 1: THERMODYNAMIC PROPERTIES     C
C                                      2: VISCOSITY                    C
C                                      3: THERMAL CONDUCTIVITY         C
C                                      4: DIELECTRIC CONSTANT          C
C IPFLG   IA   O   RESULTS OF PROPERTY BOUNDARY CHECKS                 C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 30MAY97 - AHH: CREATE FOR BACKWARD COMPATIBILITY TO VERSION 2.0      C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      DIMENSION PROPSI(NPROP), PROPR(NPROP), WAVRI(NRIMAX), RI(NRIMAX)
      DIMENSION IWANT(NPROP), IRIFLG(NRIMAX)
      DIMENSION IPCOLD(4), IPFOLD(4), IPCNEW(5), IPFNEW(5)
C
      DO 100 I=1,4
        IPCNEW(I) = IPCOLD(I)
  100 CONTINUE
      IPCNEW(5) = 0
      NRI = 0
C
      CALL PROPS(IWANT, TK, RHO, PROPSI, PROPR, I2PHCK, I2PH,
     >           ISCHK, ISFLG, ICCHK, ICFLG, IPCNEW, IPFNEW, 0,
     >           NRI, WAVRI, RI, IRIFLG)
C
      DO 200 I=1,4
        IPFOLD(I) = IPFNEW(I)
  200 CONTINUE
C
      RETURN
      END
