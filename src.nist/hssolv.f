C
      SUBROUTINE HSSOLV(MODE, TP, HS, TPOUT, D1, DV, DL, I2PH, Q, 
     >                  IWORK, PROPR, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** HSSOLV ***                                   C
C THIS ROUTINE SOLVES FOR THE CORRESPONDING T OR P FOR SPECIFIED       C
C P/H, P/S, OR T/S.  IT ALSO RETURNS THE DENSITY AT THE SOLUTION (OR   C
C DENSITIES FOR 2-PHASE SOLUTION) AND QUALITY.  THE ROUTINES THAT IT   C
C CALLS GET PASSED LIMITS WHICH ARE CHECKED.                           C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C MODE    I    I   1 = GIVEN P/H, SOLVE FOR T                          C
C                  2 = GIVEN P/S, SOLVE FOR T                          C
C                  3 = GIVEN T/S, SOLVE FOR P                          C
C TP      R    I   DEPENDING ON MODE, EITHER T(K) OR P(MPA)            C
C HS      R    I   DEPENDING ON MODE, EITHER H(KJ/KG) OR S (KJ/KG.K)   C
C TPOUT   R    O   SOLUTION FOR T(K) OR P(MPA)                         C
C D1      R    O   DENSITY (KG/M3) IF SOLUTION IN 1-PHASE REGION       C
C DV      R    O   VAPOR DENSITY (KG/M3) IF SOLUTION IN 2-PHASE REGION C
C DL      R    O   LIQ. DENSITY (KG/M3) IF SOLUTION IN 2-PHASE REGION  C
C I2PH    I    O   OUTPUT FLAG FOR RELATION TO 2-PHASE REGION          C
C                  0 = NO SOLUTION FOUND                               C
C                 -1 = SUBCOOLED LIQUID (ALSO T<TC AND P>PC)           C
C                  1 = SUPERHEATED VAPOR (ALSO T>TC)                   C
C                  2 = SOLUTION IN TWO-PHASE REGION                    C
C                  3 = UNABLE TO PERFORM 2-PHASE ANALYSIS BECAUSE P    C
C                      TOO LOW FOR SAT. CALC.  TREATED AS 1-PHASE      C
C                  4 = SAME AS 2, EXCEPT 2-PHASE ENVELOPE EXTRAPOLATED C
C                      TO REGION BELOW TRIPLE POINT                    C
C Q       R    O   QUALITY (VAPOR/TOTAL) AT SOLUTION.  IF SOLUTION IS  C
C                  1-PHASE, RETURNS 1.0 FOR SUPERCRITICAL ISOBAR OR    C
C                  ISOTHERM.  FOR 1-PHASE ON SUBCRITICAL ISOBAR OR     C
C                  ISOTHERM, RETURNS 1.0 FOR VAPOR OR 0.0 FOR LIQUID.  C  
C IWORK   IA   -   INTEGER VECTOR FOR USE BY LOWER-LEVEL ROUTINES      C
C PROPR   RA   -   VECTOR OF REDUCED PROPERTIES USED BY LOWER-LEVEL    C
C                  ROUTINES                                            C
C IERR    I    O   RETURN STATUS CODE                                  C
C                  0: CONVERGED                                        C
C                  1: NO SOLUTION INSIDE VALID RANGE                   C
C                  2: SOLUTION EXTRAPOLATED OUTSIDE RECOMMENDED RANGE  C
C                     (EXTRAPOLATION INTO FLUID REGION)                C
C                  3: UNABLE TO CONVERGE BOUNDED ROOT                  C
C                  4: SOLUTION EXTRAPOLATED OUTSIDE RECOMMENDED RANGE  C
C                     (EXTRAPOLATION INTO EQUILIBRIUM SOLID REGION)    C
C                  5: INPUT PRESSURE NOT IN VALID RANGE                C
C                  6: INPUT TEMPERATURE NOT IN VALID RANGE             C
C                  7: CALLED WITH INVALID MODE                         C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 22SEP95 - INITIAL CREATION BY AHH                                    C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 29SEP95 - AHH: CHANGE IWANT TO IWORK, REDOCUMENT ACCORDINGLY         C
C 02OCT95 - AHH: UPGRADE FOR NEW BOUNDS CHECKING IN PHSOLV             C
C 02OCT95 - AHH: ADD P/S CAPABILITY WITH PSSOLV (MODE 2)               C
C 04OCT95 - AHH: ADD T/S CAPABILITY WITH TSSOLV (MODE 3)               C
C 05OCT95 - AHH: ADD CHECKS FOR INVALID INPUT T OR P AND I2PH FLAG     C
C 19OCT95 - AHH: MOVE LIMITS INTO THEIR OWN COMMON, WLIMIT.CMN         C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C                ALSO FIX A MIN STATEMENT THAT SOME COMPILERS DISLIKE. C
C 12FEB97 - AHH: HANDLE HIGH VALUES OF T FOR T/S (WAS GETTING PMAX=0)  C
C 10JUN97 - AHH: CHANGE COMMENTS TO REFLECT CHANGES IN I2PH FLAG       C
C 23MAR99 - AHH: FIX DOCUMENTATION OF I2PH FLAG                        C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      INCLUDE 'wconst.cmn'
      INCLUDE 'wlimit.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWORK(NPROP)
      I2PH = 0
C
C CHECK MODE AND ACT ACCORDINGLY
C LIMITS ARE SAME FOR MODES 1 AND 2		    
C RECOMMENDED MAX AND MIN TEMPERATURE ARE 1273.15 AND SOLID SATURATION
C ABSOLUTE LIMITS YET TO BE DETERMINED
C
      IF ((MODE .EQ. 1) .OR. (MODE .EQ. 2)) THEN
        IF ((TP .LT. PLOWER) .OR. (TP .GT. PUPPER)) THEN
          IERR = 5
          TPOUT = 0.D0
          RETURN
        ENDIF
        TMAX = TUPPER / TCW
        IF (TP .GT. PTRIPW) THEN
          CALL TMELT(T1, TP, IFORM, IER1)
          TMIN = (T1 - 20.D0) / TCW
        ELSE
          CALL TSUB(T1, TP, IER1)
          TMIN = (T1 - 10.D0) / TCW
        ENDIF
        TRECMN = T1 / TCW
        TRECMN = MAX(TRECMN, TMIN)
        TRECMX = 1273.15D0 / TCW
        PR = TP / PCW
      ENDIF
C
      IF (MODE .EQ. 1) THEN
C
C P AND H GIVEN, SOLVE FOR T BY CALLING PHSOLV
C
        HTR = HS / RW / TCW
        CALL PHSOLV(PR, HTR, TR, TMAX, TMIN, TRECMX, TRECMN, D1R, DVR,
     >              DLR, I2PH, Q, IWORK, PROPR, IERR)
        TPOUT = TR*TCW
        IF ((IERR .EQ. 2) .AND. (TR .LT. TRECMN)) IERR=4
C
      ELSE IF (MODE .EQ. 2) THEN
C
C P AND S GIVEN, SOLVE FOR T BY CALLING PSSOLV
C
        SR = HS / RW
        CALL PSSOLV(PR, SR, TR, TMAX, TMIN, TRECMX, TRECMN, D1R, DVR,
     >              DLR, I2PH, Q, IWORK, PROPR, IERR)
        TPOUT = TR*TCW
        IF ((IERR .EQ. 2) .AND. (TR .LT. TRECMN)) IERR=4
C
      ELSE IF (MODE .EQ. 3) THEN
C
C T AND S GIVEN, SOLVE FOR P BY CALLING TSSOLV
C
        IF ((TP .LT. TLOWER) .OR. (TP .GT. TUPPER)) THEN
          IERR = 6
          TPOUT = 0.D0
          RETURN
        ENDIF
C
C FIRST, SET LIMITS DEPENDING ON TEMPERATURE
C IF T >= TRIPLE, HAVE ONLY AN UPPER BOUND FROM MELTING CURVE
C BUT FOR T ABOVE HIGHEST EXTENT OF PMELT2, SET 1000 MPA AS LIMIT
C IF BELOW LOWEST MELTING POINT (251.165), HAVE ONLY AN UPPER BOUND
C FROM THE SUBLIMATION CURVE.  IN BETWEEN IS TRICKY - SEE BELOW
C HAVE TO SET ARTIFICIAL LOWER BOUNDS, BUT THEY ARE TOO LOW TO BE
C REACHED IN ANY REAL PROCESS.
C
        PMIN = 1.D-20
        PRECMN = 1.D-20
        TR = TP / TCW
        SR = HS / RW
        IF (TP .GE. TTRIPW) THEN
          CALL PMELT2(TP, PSOLID, IFORM, IERS)
          IF (IERS .EQ. 2) THEN
            PRECMX = 1000.D0
          ELSE
            PRECMX = MIN(1000.D0, PSOLID)
            PSOLID = PSOLID / PCW
          ENDIF
          PRECMX = PRECMX / PCW
          PMAX = 2.D0*PRECMX
          CALL TSSOLV(PR, SR, TR, PMAX, PMIN, PRECMX, PRECMN, D1R, DVR,
     >                DLR, I2PH, Q, IWORK, PROPR, IERR)
          TPOUT = PR * PCW
          IF ((IERR .EQ. 2) .AND. (IERS .EQ. 0) .AND. 
     >        (PR .GT. PSOLID)) IERR = 4
        ELSE IF (TP .LT. TMELTX) THEN
          CALL PSUB(TP, PSOLID, IERR)
          PSOLID = PSOLID / PCW
          PRECMX = PSOLID
          PMAX = 1.5D0*PRECMX
          CALL TSSOLV(PR, SR, TR, PMAX, PMIN, PRECMX, PRECMN, D1R, DVR,
     >                DLR, I2PH, Q, IWORK, PROPR, IERR)
          TPOUT = PR * PCW
          IF ((IERR .EQ. 2) .AND. (PR .GT. PSOLID)) IERR = 4
        ELSE
C
C IN THIS REGION, THERE ARE 2 SEPARATE REGIONS TO BE SEARCHED.  ONE
C GOES FROM ZERO PRESSURE TO THE SUBLIMATION CURVE, THE OTHER GOES
C FROM THE ICE-I MELTING CURVE UP TO THE HIGH-PRESSURE ICE MELTING
C CURVE.  SO SEARCH EACH OF THESE SEPARATELY, MAKING SURE YOU DON'T
C OVERLAP THE SEARCHES. SEARCH MAY ALSO PICK UP METASTABLE STATES OR
C EXTRAPOLATED V/L COEXISTENCE REGION.
C
          CALL PSUB(TP, PSUBL, IER1)
          CALL PMELT1(TP, PMLOW, IER2)
          CALL PMELT2(TP, PMHIGH, IFORM, IER3)
C
C FIRST LOOK FOR ROOT IN VAPOR REGION
C
          PRECMX = PSUBL / PCW
          PMAX = (MIN(1.5*PSUBL, 0.5*(PSUBL+PMLOW))) / PCW 
          CALL TSSOLV(PR, SR, TR, PMAX, PMIN, PRECMX, PRECMN, D1R, DVR,
     >                DLR, I2PH, Q, IWORK, PROPR, IERR)
          IF ((IERR .EQ. 0) .OR. (IERR .EQ. 2)) THEN
            TPOUT = PR * PCW
            IF ((IERR .EQ. 2) .AND. (PR .GT. PRECMX)) IERR = 4
          ELSE
C
C IF NO VAPOR ROOT, LOOK FOR A LIQUID ROOT
C  
            PRECMN = PMLOW / PCW
            PMIN = (MAX(PMLOW/1.5, 0.5*(PSUBL+PMLOW))) / PCW
            PRECMX = PMHIGH / PCW
            PMAX = 1.5*PRECMX
            CALL TSSOLV(PR, SR, TR, PMAX, PMIN, PRECMX, PRECMN, D1R,
     >                  DVR, DLR, I2PH, Q, IWORK, PROPR, IERR)
            TPOUT = PR * PCW
            IF (IERR .EQ. 2) IERR = 4
          ENDIF
        ENDIF
      ELSE
        IERR = 7
        RETURN
      ENDIF
      D1 = D1R*RHOCW
      DV = DVR*RHOCW
      DL = DLR*RHOCW
      RETURN       
      END
