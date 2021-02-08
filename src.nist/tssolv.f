C
      SUBROUTINE TSSOLV(PR, SR, TR, PMAX, PMIN, PRECMX, PRECMN, D1R,
     >                  DVR, DLR, I2PH, Q, IWANT, PROPR, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** TSSOLV ***                                   C
C THIS ROUTINE FINDS THE REDUCED PRESSURE CORRESPONDING TO A GIVEN     C
C REDUCED TEMPERATURE PR AND REDUCED ENTROPY SR.                       C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C PR      R    O   REDUCED (P/PC) PRESSURE                             C
C SR      R    I   REDUCED ENTROPY S/R                                 C
C TR      R    I   REDUCED TEMPERATURE, T/TC                           C
C PMAX    R    I   MAXIMUM VALID REDUCED PRESSURE                      C
C PMIN    R    I   MINIMUM VALID REDUCED PRESSURE                      C
C PRECMX  R    I   MAXIMUM RECOMMENDED REDUCED PRESSURE                C
C PRECMN  R    I   MINIMUM RECOMMENDED REDUCED PRESSURE                C
C D1R     R    O   REDUCED DENSITY IF SOLUTION IN 1-PHASE REGION       C
C DVR     R    O   REDUCED VAPOR DENSITY IF SOLUTION IN 2-PHASE REGION C
C DLR     R    O   REDUCED LIQ. DENSITY IF SOLUTION IN 2-PHASE REGION  C
C I2PH    I    O   OUTPUT FLAG FOR RELATION TO 2-PHASE REGION          C
C                  0 = NO SOLUTION FOUND                               C
C                 -1 = COMPRESSED LIQUID (ALSO T<TC AND P>PC)          C
C                  1 = SUPERHEATED/EXPANDED VAPOR (ALSO T>TC)          C
C                  2 = SOLUTION IN TWO-PHASE REGION                    C
C                  3 = UNABLE TO PERFORM 2-PHASE ANALYSIS BECAUSE T    C
C                      TOO LOW FOR SAT. CALC.  TREATED AS 1-PHASE      C
C                  4 = SAME AS 2, EXCEPT 2-PHASE ENVELOPE EXTRAPOLATED C
C                      TO REGION BELOW TRIPLE POINT                    C
C Q       R    O   QUALITY (VAPOR/TOTAL) AT SOLUTION.  IF SOLUTION IS  C
C                  1-PHASE, RETURNS 1.0 FOR SUPERCRITICAL ISOTHERM.    C
C                  FOR 1-PHASE ON SUBCRITICAL ISOTHERM, RETURNS 1.0    C
C                  FOR VAPOR OR 0.0 FOR LIQUID.                        C
C IWANT   IA   -   PROPERTY REQUEST VECTOR FOR USE IN PROPS2 ROUTINE   C
C PROPR   RA   -   VECTOR OF REDUCED PROPERTIES AS REQUESTED BY IWANT  C
C IERR    I    O   RETURN STATUS CODE                                  C
C                  0: CONVERGED                                        C
C                  1: NO SOLUTION INSIDE VALID PRESSURE RANGE          C
C                  2: SOLUTION VALID, BUT OUTSIDE RECOMMENDED REGION   C
C                  3: UNABLE TO CONVERGE BOUNDED ROOT                  C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 03OCT95 - INITIAL CREATION BY AHH (FROM PSSOLV)                      C
C 05OCT95 - AHH: PUT IN I2PH FLAG FOR RELATION TO 2-PHASE REGION       C
C 11DEC95 - AHH: RENUMBERING OF PROPERTY ARRAY                         C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 08JAN97 - AHH: CHANGE IN ARGUMENT LIST OF PROPS2                     C
C 04JUN97 - AHH: ALLOW FOR I2PH=4                                      C
C 23MAR99 - AHH: CORRECTLY SET I2PH FOR T>TC                           C
C 26FEB04 - AHH: IMPROVE CONVERGENCE IN LIQUID REGION BY USING P (WHICHC
C                BEHAVES MORE LINEARLY) INSTEAD OF LOG(P) IN ITERATION C
C 27FEB04 - AHH: TIGHTEN CONVERGENCE TOLERANCES BY FACTOR OF 10        C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      INCLUDE 'wconst.cmn'
      INCLUDE 'wlimit.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWANT(NPROP)
      PARAMETER (RTOL = 1.0D-9, ATOL=1.D-8)
      TBOT = TSMIN / TCW
      RTW = RW * TR * TCW
      I2PH = 0
      D1R = 0.D0
      DVR = 0.D0
      DLR = 0.D0
      PHIGH = PMAX
      PLOW = PMIN
C
C FIRST, EVALUATE SR AT PMAX AND PMIN TO SEE IF VALUE IS WITHIN BOUNDS
C IF YOU CAN'T SOLVE AT THOSE BOUNDS, TRY AT THE RECOMMENDED BOUNDS
C
      CALL DFIND0(DMAX, PMAX, TR, DPDR, IWANT, PROPR, IERR)
      IF (IERR .NE. 0) THEN
        CALL DFIND0(DMAX, PRECMX, TR, DPDR, IWANT, PROPR, IERR)
        PHIGH = PRECMX
      ENDIF        
      CALL IVZERO(IWANT, NPROP)
      IWANT(7) = 1
      CALL PROPS2(IWANT, 0, TR, DMAX, PROPR)
      SMAX = PROPR(7)
      IF (SR .LT. SMAX) THEN
        IERR = 1
        PR = PHIGH
        Q = 1.D0
        D1R = DMAX
        RETURN
      ENDIF
      CALL DFIND0(DMIN, PMIN, TR, DPDR, IWANT, PROPR, IERR)
      IF (IERR .NE. 0) THEN
        CALL DFIND0(DMIN, PRECMN, TR, DPDR, IWANT, PROPR, IERR)
        PLOW = PRECMN
      ENDIF        
      CALL IVZERO(IWANT, NPROP)
      IWANT(7) = 1
      CALL PROPS2(IWANT, 0, TR, DMIN, PROPR)
      SMIN = PROPR(7)
      IF (SR .GT. SMIN) THEN
        IERR = 1
        PR = PLOW
        Q = 0.D0
        D1R = DMIN
        RETURN
      ENDIF
C
C NOW THAT YOU HAVE ESTABLISHED THAT IT IS WITHIN BOUNDS, BRANCH DEPENDING
C ON WHETHER THE TEMPERATURE ALLOWS A SATURATION CALCULATION
C
      IF ((TR .GE. 1.D0) .OR. (TR .LT. TBOT)) THEN
C
C TEMPERATURE ABOVE TC, SO DON'T HAVE TO WORRY ABOUT PHASE TRANSITION.
C ALREADY HAVE UPPER AND LOWER BOUND, SO DO BOUNDED SECANT RIGHT AWAY.
C ALSO DON'T LOOK FOR V/L COEXISTENCE IF TEMPERATURE BELOW MINIMUM FOR
C SATURATION CALCULATION.
C
        SLOW = SMIN
        SHIGH = SMAX  
        Q = 1.D0
        IF (TR .LT. TBOT) I2PH = 3
        GO TO 222
      ELSE
C
C CHECK SATURATION VALUES
C
        CALL PCOEX(TR, PR, DLR, DVR, IWANT, PROPR, IERC)
        CALL IVZERO(IWANT, NPROP)
        IWANT(7) = 1
        CALL PROPS2(IWANT, 0, TR, DVR, PROPR)
        SVAP = PROPR(7)
        CALL PROPS2(IWANT, 0, TR, DLR, PROPR)
        SLIQ = PROPR(7)
        IF (SR .GE. SVAP) THEN
C
C ONE-PHASE VAPOR, BOUNDS ARE SAT. VAPOR AND PMIN
C
          I2PH = 1
          SHIGH = SVAP
          SLOW = SMIN
          PHIGH = PR
          Q = 1.D0
          GO TO 222
        ELSE IF (SR .LE. SLIQ) THEN
C
C ONE-PHASE LIQUID, BOUNDS ARE SAT. LIQUID AND PMAX
C
          I2PH = -1
          SLOW = SLIQ
          SHIGH = SMAX
          PLOW = PR
          Q = 0.D0
C USE DIFFERENT CONVERGENCE VARIABLE FOR LIQUID NOT CLOSE TO TC
          IF (TR .LT. 0.99D0) GO TO 333
          GO TO 222
        ELSE
C
C IN TWO-PHASE REGION, CALCULATE QUALITY AND RETURN
C
          IF (IERC .EQ. 4) THEN
            I2PH = 4
          ELSE
            I2PH = 2
          ENDIF
          Q = (SR-SLIQ) / (SVAP-SLIQ)
          IERR = 0
          RETURN
        ENDIF
      ENDIF
C    
  222 CONTINUE
C
C SOLUTION IS NOW BRACKETED, USE BOUNDED SECANT TO SOLVE
C WITH LOG PRESSURE AS VARIABLE
C
      PLOWL = LOG(PLOW)
      PHIGHL = LOG(PHIGH)
      FLOW = SR - SLOW
      FHIGH = SR - SHIGH
      DO 300 I=1,200
        FRDIST = -FLOW / (FHIGH-FLOW)
        IF (FRDIST .LT. 0.01d0) FRDIST = 0.01d0
        IF (FRDIST .GT. 0.99d0) FRDIST = 0.99d0
        PNEWL = PLOWL + FRDIST*(PHIGHL-PLOWL)
        PNEW = DEXP(PNEWL)
        CALL DFIND0(DNEW, PNEW, TR, DPDR, IWANT, PROPR, IERR)
        CALL IVZERO(IWANT, NPROP)
        IWANT(7) = 1
        CALL PROPS2(IWANT, 0, TR, DNEW, PROPR)
        SNEW = PROPR(7)
C
C CHECK FOR CONVERGENCE
C
        IF (DABS(SR) .GT. ATOL) THEN
          IF (DABS(1.D0-SNEW/SR) .LT. RTOL) GO TO 999
        ENDIF
        IF(DABS(SNEW-SR) .LT. ATOL) GO TO 999
C
C MAKE NEW BOUNDS DEPENDING ON WHETHER NEW GUESS WAS HIGH OR LOW
C
        FNEW = SR - SNEW
        IF (FNEW .GE. 0.D0) THEN
          FHIGH = FNEW
          PHIGHL = PNEWL
        ELSE
          FLOW = FNEW
          PLOWL = PNEWL
        ENDIF
  300 CONTINUE
C FAILURE TO CONVERGE
      GO TO 444
C
  333 CONTINUE
C
C SOLUTION IS NOW BRACKETED, USE BOUNDED SECANT TO SOLVE
C WITH PRESSURE AS VARIABLE FOR COMPRESSED LIQUID
C
      FLOW = SR - SLOW
      FHIGH = SR - SHIGH
      DO 400 I=1,200
        FRDIST = -FLOW / (FHIGH-FLOW)
        IF (FRDIST .LT. 0.01d0) FRDIST = 0.01d0
        IF (FRDIST .GT. 0.99d0) FRDIST = 0.99d0
        PNEW = PLOW + FRDIST*(PHIGH-PLOW)
        CALL DFIND0(DNEW, PNEW, TR, DPDR, IWANT, PROPR, IERR)
        CALL IVZERO(IWANT, NPROP)
        IWANT(7) = 1
        CALL PROPS2(IWANT, 0, TR, DNEW, PROPR)
        SNEW = PROPR(7)
C
C CHECK FOR CONVERGENCE
C
        IF (DABS(SR) .GT. ATOL) THEN
          IF (DABS(1.D0-SNEW/SR) .LT. RTOL) GO TO 999
        ENDIF
        IF(DABS(SNEW-SR) .LT. ATOL) GO TO 999
C
C MAKE NEW BOUNDS DEPENDING ON WHETHER NEW GUESS WAS HIGH OR LOW
C
        FNEW = SR - SNEW
        IF (FNEW .GE. 0.D0) THEN
          FHIGH = FNEW
          PHIGH = PNEW
        ELSE
          FLOW = FNEW
          PLOW = PNEW
        ENDIF
  400 CONTINUE
C
C FAILURE TO CONVERGE
C
  444 CONTINUE
      IERR = 3
      PR = PNEW
      D1R = DNEW
      RETURN
C
  999 CONTINUE
C
C CONVERGENCE
C
C CHECK AGAINST "RECOMMENDED" BOUNDS
C
      PR = PROPR(2)*RTW*DNEW*RHOCW*1.D-3/PCW
      IF ((PR .LT. PRECMN) .OR. (PR .GT. PRECMX)) THEN
        IERR = 2
      ELSE
        IERR = 0
      ENDIF
      D1R = DNEW
      IF (TR .GE. 1.D0) THEN
        I2PH = 1
      ENDIF
      RETURN       
      END
