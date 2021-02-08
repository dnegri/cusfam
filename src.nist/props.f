C
      SUBROUTINE PROPS(IWANT, TK, RHO, PROPSI, PROPR, I2PHCK, I2PH,
     >                  ISCHK, ISFLG, ICCHK, ICFLG, IPCHK, IPFLG, IGFLG,
     >                  NRI, WAVRI, RI, IRIFLG)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PROPS ***                                    C
C THIS ROUTINE COMPUTES ALL PROPERTIES REQUESTED VIA THE IWANT VECTOR  C
C AT A GIVEN TEMPERATURE AND DENSITY.  PROPERTIES ARE RETURNED IN SI   C
C UNITS: KG, M3, K, MPA, KJ.  THE LOWER-LEVEL ROUTINE PROPS1 IS        C
C CALLED TO RETURN THE PROPERTIES.  IT ALSO CHECKS TO SEE IF THE POINT C
C IS IN THE 2-PHASE REGION AND RETURNS THE APPROPRIATE QUALITY AND     C
C BULK PROPERTIES.  BOUNDS CHECKING IS ALSO DONE HERE IF REQUESTED     C
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
C                                      5: REFRACTIVE INDEX             C
C IPFLG   IA   O   RESULTS OF PROPERTY BOUNDARY CHECKS                 C
C IGFLG   I    I   WHETHER TO COMPUTE AND RETURN ONLY IDEAL-GAS PROPS  C
C                  1=IDEAL GAS, 0=REAL FLUID                           C
C NRI     I    I   NUMBER OF REFRACTIVE INDEX POINTS TO COMPUTE        C
C WAVRI   RA   I   ARRAY OF WAVELENGTHS (IN MICROMETERS) AT WHICH TO   C
C                  COMPUTE REFRACTIVE INDEX                            C
C RI      RA   O   RETURNED ARRAY OF REFRACTIVE INDICES                C
C IRIFLG  IA   O   ARRAY OF RETURN FLAGS FOR REF. INDEX WAVELENGTHS    C
C                  0     = OK                                          C
C                  1(-1) = LARGER(SMALLER) THAN RECOMMENDED RANGE,     C
C                          VALUE IS AN EXTRAPOLATION                   C
C                  2(-2) = TOO LARGE(SMALL), BEYOND REASONABLE         C
C                          EXTRAPOLATION RANGE. WILL STILL RETURN      C
C                          COMPUTED NUMBER IF WITHIN RANGE WHERE CALC. C
C                          DOESN'T BLOW UP, OTHERWISE RETURNS ZERO.    C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 28SEP95 - INITIAL CREATION BY AHH                                    C
C 29SEP95 - UPGRADE HANDLING OF QUALITY                                C
C 05OCT95 - AHH: PUT IN BOUNDS CHECKING TYPES 2 AND 3                  C
C 27OCT95 - AHH: FIX BOUNDS CHECKING AND SETTING OF I2PH FLAG          C
C 09NOV95 - AHH: REDO BOUNDS CHECKING WITH CALL TO BNDCHK              C
C 27NOV95 - AHH: ADD DIELECTRIC CONSTANT IN SLOT 21                    C
C 07DEC95 - AHH: ADD VOLUME EXPANSIVITY AS PROPERTY 22                 C
C 07DEC95 - AHH: IMPLEMENT CALCULATION OF CV IN 2-PHASE REGION         C
C 11DEC95 - AHH: REORDER PROPERTY NUMBERING                            C
C 13DEC95 - AHH: FIX 2-PHASE CV CALCULATION                            C
C 07JUN96 - AHH: IF T<TC AND P>PC, CALL IT SUBCOOLED                   C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 17SEP96 - AHH: EXPUNGE TAB CHARACTERS                                C
C 07NOV96 - AHH: CORRECT SOME COMMENTS, PARTICULARLY ABOUT QUALITY     C
C 09JAN97 - AHH: ADD IGFLG TO ARGUMENT LIST TO COMPUTE I.G. PROPS.     C
C 13JAN97 - AHH: ALLOW V/L CALCS BELOW TRIPLE POINT                    C
C 26FEB97 - AHH: ADD DERIVATIVE PROPERTIES IN SLOTS 23-30              C
C 05MAR97 - AHH: ADD DIELECTRIC DERIVATIVES IN SLOTS 31-37             C
C 06MAR97 - AHH: ADD DEBYE-HUCKEL SLOPES IN SLOTS 38-43                C
C 25MAR97 - AHH: ADD REFRACTIVE INDEX CALCULATIONS                     C
C 20MAY97 - AHH: ADD I2PH = 4 CAPABILITY                               C
C 30MAY97 - AHH: CHANGED NAME TO PROPS TO PRESERVE OLD PROPS0 ARG LIST C
C 01JUL97 - AHH: DO BOUNDARY CHECKS IF 2-PHASE BELOW TRIPLE POINT      C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
C
C NEED CONSTANTS FOR WATER 
C
      INCLUDE 'wconst.cmn'
      DIMENSION PROPSI(NPROP), PROPR(NPROP), WAVRI(NRIMAX), RI(NRIMAX)
      DIMENSION IWANT(NPROP), IPCHK(5), IPFLG(5), IRIFLG(NRIMAX)
C
C NEED THREE TEMPORARY ARRAYS FOR VARIOUS PURPOSES
C
      DIMENSION TPROP(NPROP)
      DIMENSION IWTMP(NPROP)
      DIMENSION IPDUM(5)
C
C CHECK FOR 2-PHASE (V-L) REGION IF REQUESTED
C DON'T DO THIS CHECK IF IDEAL GAS PROPERTIES REQUESTED
C
      I2PH = 0
      IF ((I2PHCK .NE. 0) .AND. (IGFLG .NE. 1)) THEN
        IF (TK .LT. TCW) THEN
          CALL PSAT(TK, PMPA, RHOL, RHOV, IWTMP, PROPR, IERRPS)
C
C IF T TOO LOW TO CHECK V/L SATURATION CURVE, CAN'T GET COEXISTING
C DENSITIES FROM PSAT TO CHECK.  SO SET RETURN CODE AND CALC AS
C IF SINGLE PHASE
C
          IF (IERRPS .EQ. 1) THEN
            I2PH = 3
C
          ELSE IF ((RHO .LT. RHOL) .AND. (RHO .GT. RHOV)) THEN
C
C 2-PHASE, FIGURE QUALITY AND GET PROPERTIES AT PHASE BOUNDARIES
C
            IF (IERRPS .EQ. 4) THEN
              I2PH = 4
            ELSE
              I2PH = 2
            ENDIF
            Q = (1.D0/RHO-1.D0/RHOL) / (1.D0/RHOV-1.D0/RHOL)
            DO 150 I=1,NPROP
              IWTMP(I) = IWANT(I)
  150       CONTINUE
C
C IF YOU ARE GOING TO BE FIGURING 2-PHASE CV, MAKE SURE YOU CALL FOR
C ALL THE NEEDED PROPERTIES IN BOTH PHASES
C
            IF (IWANT(8) .NE. 0) THEN
              IWTMP(2) = 1
              IWTMP(6) = 1
              IWTMP(9) = 1
              IWTMP(10) = 1
              IWTMP(14) = 1
              IWTMP(15) = 1
            ENDIF
            CALL PROPS1(IWTMP, IGFLG, TK, RHOV, TPROP, PROPR)
            CALL PROPS1(IWTMP, IGFLG, TK, RHOL, PROPSI, PROPR)
C
C ZERO OUT QUANTITIES THAT ARE MEANINGLESS IN 2-PHASE REGION
C FIGURE OTHERS IF REQUESTED - THOSE THAT ARE IDENTICAL IN BOTH PHASES
C SHOULD ALREADY BE IN THE PROPSI ARRAY
C
            Q1 = 1.D0 - Q
            IF (IWANT(8) .NE. 0) THEN
C
C LENGTHY CALCULATION OF 2-PHASE CV.  FIRST, CALCULATE DPSAT/DT
C (HAVE TO TOSS IN A 1.D3 SINCE P IN MPA BUT H IN KJ)
C
              VVAP = 1.D0 / RHOV
              VLIQ = 1.D0 / RHOL
              DPSDT = (TPROP(6)-PROPSI(6)) / (TK*(VVAP-VLIQ))
              DPSDT = DPSDT / 1.D3
C
C NEXT CALCULATE THE VARIATION OF THE SATURATED VOLUMES WITH T
C
              DVVDT = VVAP * (TPROP(15) - TPROP(14)*DPSDT)
              DVLDT = VLIQ * (PROPSI(15) - PROPSI(14)*DPSDT)
C
C WE ALSO NEED THE VARIATIONS OF SATURATED INT. ENG. WITH T
C
              DUVDP = 1.D3*VVAP*(TPROP(14)*TPROP(2) - TPROP(15)*TK)
              DULDP = 1.D3*VLIQ*(PROPSI(14)*PROPSI(2) - PROPSI(15)*TK)
              DUVDT = TPROP(9) - 1.D3*TPROP(2)*VVAP*TPROP(15)
              DULDT = PROPSI(9) - 1.D3*PROPSI(2)*VLIQ*PROPSI(15)
              DUVDTS = DUVDT + DUVDP*DPSDT
              DULDTS = DULDT + DULDP*DPSDT
C
C NOW YOU CAN CALCULATE ISOCHORIC DQ/DT, WHICH THEN GOES INTO CV
C
              DQDT = -(Q*DVVDT + Q1*DVLDT) / (VVAP - VLIQ)
              DELU = TPROP(10) - PROPSI(10)
              CV2PH = Q*DUVDTS + Q1*DULDTS + DQDT*DELU
              PROPSI(8) = CV2PH
            ENDIF 
            IF (IWANT(3) .NE. 0) THEN
              PROPSI(3) = RHO
            ENDIF
            IF (IWANT(4) .NE. 0) THEN
              PROPSI(4) = 1.D0/RHO
            ENDIF
            IF (IWANT(6) .NE. 0) THEN
              PROPSI(6) = Q*TPROP(6) + Q1*PROPSI(6)
            ENDIF
            IF (IWANT(7) .NE. 0) THEN
              PROPSI(7) = Q*TPROP(7) + Q1*PROPSI(7)
            ENDIF
            IF (IWANT(10) .NE. 0) THEN
              PROPSI(10) = Q*TPROP(10) + Q1*PROPSI(10)
            ENDIF
            IF (IWANT(11) .NE. 0) THEN
              PROPSI(11) = Q*TPROP(11) + Q1*PROPSI(11)
            ENDIF
            PROPSI(9) = 0.D0
            DO 210 I=14,43
              PROPSI(I) = 0.D0
  210       CONTINUE
            IF (NRI .GT. 0) THEN
              DO 212 I=1,NRI
                RI(I) = 0.D0
  212         CONTINUE
            ENDIF
C
C BOUNDS CHECKING - NEED TO DO IF IN 2-PHASE REGION BELOW TRIPLE PT.
C ELSE, DO BOUNDS CHECKING ONLY FOR NEAR-CRITICAL (SINCE OTHER
C BOUNDS CAN'T BE VIOLATED IN 2-PHASE REGION)
C
            IF (I2PH .EQ. 4) THEN
              CALL BNDCHK(TK, PDUM, RHOV, 2, 0, ISFLG, 0, ICFLG,
     >                    IPCHK, IPFLG, IWTMP, TPROP, PROPR)
              CALL BNDCHK(TK, PDUM, RHOL, 2, 0, ISFLG, 0, ICFLG,
     >                    IPCHK, IPDUM, IWTMP, TPROP, PROPR)
              DO 105 I=1,5
                IPFLG(I) = MAX(IPFLG(I), IPDUM(I))
  105         CONTINUE
            ELSE IF (ICCHK .EQ. 1) THEN
              DO 110 I=1,5
                IPDUM(I) = 0
  110         CONTINUE
              CALL BNDCHK(TK, PDUM, RHOV, 2, 0, ISFLG, ICCHK, ICVAP,
     >                    IPDUM, IPFLG, IWTMP, TPROP, PROPR)
              CALL BNDCHK(TK, PDUM, RHOL, 2, 0, ISFLG, ICCHK, ICLIQ,
     >                    IPDUM, IPFLG, IWTMP, TPROP, PROPR)
              ICFLG = MAX(ICVAP, ICLIQ)
            ENDIF
C
C IF ASKED FOR SOLIDS CHECKING, ALSO SET THAT FLAG IF IN 2-PHASE
C REGION BELOW TRIPLE-POINT TEMPERATURE
C
            IF ((ISCHK.EQ.1) .AND. (I2PH.EQ.4)) ISFLG = 1
C
          ENDIF
        ELSE
C
C ABOVE TC, SO CALL IT SUPERHEATED
C
          I2PH = 1
        ENDIF
      ENDIF
C
C IF YOU DIDN'T DO THE 2-PHASE CALCULATION, CALL PROPS1 DIRECTLY
C BUT FIRST CHECK BOUNDS, AND DON'T CALL IF OUT OF BOUNDS
C
      IF ((I2PH .NE. 2) .AND. (I2PH .NE. 4)) THEN
        Q = 0.D0
        CALL BNDCHK(TK, PDUM, RHO, 2, ISCHK, ISFLG, ICCHK, ICFLG,
     >              IPCHK, IPFLG, IWTMP, TPROP, PROPR)
        IPMAX = 0
        DO 120 I=1,5
          IPMAX = MAX(IPMAX, IPFLG(I))
  120   CONTINUE
        IPMAX = MAX(IPMAX, ISFLG)
        IF (IPMAX .LT. 2) THEN
          CALL PROPS1(IWANT, IGFLG, TK, RHO, PROPSI, PROPR)
          IF (NRI .GT. 0) THEN
            CALL RIND(TK, RHO, WAVRI, RI, IRIFLG, NRI)
          ENDIF
        ENDIF
      ENDIF
C
C IF REQUESTED, CALCULATED WHETHER SUPERHEATED, SUBCOOLED, ETC.
C IF T>TC, WAS ALREADY SET I2PH=1 AT BOTTOM OF 2-PHASE CHECKING
C
      IF ((I2PHCK .NE. 0) .AND. (I2PH .EQ. 0)) THEN
        IF ((PROPR(2) .LT. 1.D0) .AND. (RHO .LE. RHOV)) THEN
          I2PH = 1
        ELSE
          I2PH = -1
        ENDIF
      ENDIF
C
      PROPSI(5) = Q
      RETURN
      END
