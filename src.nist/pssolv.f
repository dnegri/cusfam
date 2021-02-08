C
      SUBROUTINE PSSOLV(PR, SR, TR, TMAX, TMIN, TRECMX, TRECMN, D1R,
     >                  DVR, DLR, I2PH, Q, IWANT, PROPR, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PSSOLV ***                                   C
C THIS ROUTINE FINDS THE REDUCED TEMPERATURE CORRESPONDING TO A GIVEN  C
C REDUCED PRESSURE PR AND REDUCED ENTROPY SR.                          C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C PR      R    I   REDUCED (P/PC) PRESSURE                             C
C SR      R    I   REDUCED ENTROPY S/R                                 C
C TR      R    O   REDUCED TEMPERATURE, T/TC                           C
C TMAX    R    I   MAXIMUM VALID REDUCED TEMPERATURE                   C
C TMIN    R    I   MINIMUM VALID REDUCED TEMPERATURE                   C
C TRECMX  R    I   MAXIMUM RECOMMENDED REDUCED TEMPERATURE             C
C TRECMN  R    I   MINIMUM RECOMMENDED REDUCED TEMPERATURE             C
C D1R     R    O   REDUCED DENSITY IF SOLUTION IN 1-PHASE REGION       C
C DVR     R    O   REDUCED VAPOR DENSITY IF SOLUTION IN 2-PHASE REGION C
C DLR     R    O   REDUCED LIQ. DENSITY IF SOLUTION IN 2-PHASE REGION  C
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
C                  1-PHASE, RETURNS 1.0 FOR SUPERCRITICAL ISOBAR.  FOR C
C                  1-PHASE ON SUBCRITICAL ISOBAR, RETURNS 1.0 FOR VAPORC
C                  OR 0.0 FOR LIQUID.                                  C  
C IWANT   IA   -   PROPERTY REQUEST VECTOR FOR USE IN PROPS2 ROUTINE   C
C PROPR   RA   -   VECTOR OF REDUCED PROPERTIES AS REQUESTED BY IWANT  C
C IERR    I    O   RETURN STATUS CODE                                  C
C                  0: CONVERGED                                        C
C                  1: NO SOLUTION INSIDE VALID TEMPERATURE RANGE       C
C                  2: SOLUTION VALID, BUT OUTSIDE RECOMMENDED REGION   C
C                  3: UNABLE TO CONVERGE BOUNDED ROOT                  C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 02OCT95 - INITIAL CREATION BY AHH (FROM PHSOLV)                      C
C 05OCT95 - AHH: PUT IN I2PH FLAG FOR RELATION TO 2-PHASE REGION       C
C 11DEC95 - AHH: RENUMBERING OF PROPERTY ARRAY                         C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 08JAN97 - AHH: CHANGE IN ARGUMENT LIST OF PROPS2                     C
C 04JUN97 - AHH: ALLOW FOR I2PH=4                                      C
C 23MAR99 - AHH: CORRECTLY SET I2PH FOR P>PC                           C
C 26FEB04 - AHH: TIGHTEN CONVERGENCE TOLERANCES                        C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      INCLUDE 'wconst.cmn'
      INCLUDE 'wlimit.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWANT(NPROP)
      PARAMETER (RTOL = 1.0D-9, ATOL=1.D-8)
      PBOT = PSMIN / PCW
      I2PH = 0
      D1R = 0.D0
      DVR = 0.D0
      DLR = 0.D0
      THIGH = TMAX
      TLOW = TMIN
C
C FIRST, EVALUATE SR AT TMAX AND TMIN TO SEE IF VALUE IS WITHIN BOUNDS
C IF YOU CAN'T SOLVE AT THOSE BOUNDS, TRY AT THE RECOMMENDED BOUNDS
C
      CALL DFIND0(DMAX, PR, TMAX, DPDR, IWANT, PROPR, IERR)
      IF (IERR .NE. 0) THEN
        CALL DFIND0(DMAX, PR, TRECMX, DPDR, IWANT, PROPR, IERR)
        THIGH = TRECMX
      ENDIF        
      CALL IVZERO(IWANT, NPROP)
      IWANT(7) = 1
      CALL PROPS2(IWANT, 0, THIGH, DMAX, PROPR)
      SMAX = PROPR(7)
      IF (SR .GT. SMAX) THEN
        IERR = 1
        TR = THIGH
        Q = 1.D0
        D1R = DMAX
        RETURN
      ENDIF
      CALL DFIND0(DMIN, PR, TMIN, DPDR, IWANT, PROPR, IERR)
      IF (IERR .NE. 0) THEN
        CALL DFIND0(DMIN, PR, TRECMN, DPDR, IWANT, PROPR, IERR)
        TLOW = TRECMN
      ENDIF        
      CALL IVZERO(IWANT, NPROP)
      IWANT(7) = 1
      CALL PROPS2(IWANT, 0, TLOW, DMIN, PROPR)
      SMIN = PROPR(7)
      IF (SR .LT. SMIN) THEN
        IERR = 1
        TR = TLOW
        Q = 0.D0
        D1R = DMIN
        RETURN
      ENDIF
C
C NOW THAT YOU HAVE ESTABLISHED THAT IT IS WITHIN BOUNDS, BRANCH DEPENDING
C ON WHETHER THE PRESSURE ALLOWS A SATURATION CALCULATION
C
      IF ((PR .GE. 1.D0) .OR. (PR .LT. PBOT)) THEN
C
C PRESSURE ABOVE PC, SO DON'T HAVE TO WORRY ABOUT PHASE TRANSITION.
C ALREADY HAVE UPPER AND LOWER BOUND, SO DO BOUNDED SECANT RIGHT AWAY.
C ALSO DON'T LOOK FOR V/L COEXISTENCE IF PRESSURE BELOW MINIMUM FOR
C SATURATION CALCULATION.
C
        SLOW = SMIN
        SHIGH = SMAX  
        Q = 1.D0
        IF (PR .LT. PBOT) I2PH = 3
        GO TO 222
      ELSE
C
C CHECK SATURATION VALUES
C
        CALL TCOEX(PR, TR, DLR, DVR, IWANT, PROPR, IERC)
        CALL IVZERO(IWANT, NPROP)
        IWANT(7) = 1
        CALL PROPS2(IWANT, 0, TR, DVR, PROPR)
        SVAP = PROPR(7)
        CALL PROPS2(IWANT, 0, TR, DLR, PROPR)
        SLIQ = PROPR(7)
        IF (SR .GE. SVAP) THEN
C
C ONE-PHASE VAPOR, BOUNDS ARE SAT. VAPOR AND THIGH
C
          I2PH = 1
          TLOW = TR
          SLOW = SVAP
          SHIGH = SMAX
          Q = 1.D0
          GO TO 222
        ELSE IF (SR .LE. SLIQ) THEN
C
C ONE-PHASE LIQUID, BOUNDS ARE SAT. LIQUID AND TMIN
C
          I2PH = -1
          THIGH = TR
          SLOW = SMIN
          SHIGH = SLIQ
          Q = 0.D0
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
C
      FLOW = SLOW - SR
      FHIGH = SHIGH - SR
      DO 300 I=1,100
        FRDIST = -FLOW / (FHIGH-FLOW)
        IF (FRDIST .LT. 0.01) FRDIST = 0.01
        IF (FRDIST .GT. 0.99) FRDIST = 0.99
        TNEW = TLOW + FRDIST*(THIGH-TLOW)
        CALL DFIND0(DNEW, PR, TNEW, DPDR, IWANT, PROPR, IERR)
        CALL IVZERO(IWANT, NPROP)
        IWANT(7) = 1
        CALL PROPS2(IWANT, 0, TNEW, DNEW, PROPR)
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
        FNEW = SNEW - SR
        IF (FNEW .GE. 0.D0) THEN
          FHIGH = FNEW
          THIGH = TNEW
        ELSE
          FLOW = FNEW
          TLOW = TNEW
        ENDIF
  300 CONTINUE
C
C FAILURE TO CONVERGE
C
      IERR = 3
      TR = TNEW
      D1R = DNEW
      RETURN
C
  999 CONTINUE
C
C CONVERGENCE
C
C CHECK AGAINST "RECOMMENDED" BOUNDS
C
      IF ((TNEW .LT. TRECMN) .OR. (TNEW .GT. TRECMX)) THEN
        IERR = 2
      ELSE
        IERR = 0
      ENDIF
      TR = TNEW
      D1R = DNEW
      IF (PR .GE. 1.D0) THEN
        IF (TR .GE. 1.D0) THEN
          I2PH = 1
        ELSE
          I2PH = -1
        ENDIF
      ENDIF
      RETURN       
      END
