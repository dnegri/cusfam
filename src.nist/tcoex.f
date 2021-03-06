C
      SUBROUTINE TCOEX(PR, TR, RHOL, RHOV, IWANT, PROPR, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** TCOEX ***                                    C
C THIS ROUTINE COMPUTES THE REDUCED COEXISTENCE TEMPERATURE T/TC AT A  C
C GIVEN REDUCED PRESSURE PR.  IT ALSO RETURNS THE COEXISTING           C
C DENSITIES.  THE STRATEGY IS TO GET A GOOD INITIAL GUESS AND THEN     C
C CONVERGE THE LN OF THE K-FACTOR TO ZERO.                             C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C PR      R    I   REDUCED PRESSURE, P/PC                              C
C TR      R    O   REDUCED COEXISTENCE TEMPERATURE, T/TC               C
C RHOL    R    O   REDUCED SAT. LIQUID DENSITY, RHOL/RHOC              C
C RHOV    R    O   REDUCED SAT. VAPOR DENSITY, RHOV/RHOC               C
C IWANT   IA   -   PROPERTY REQUEST VECTOR FOR USE IN PROPS2 ROUTINE   C
C PROPR   RA   O   VECTOR OF REDUCED PROPERTIES AS REQUESTED BY IWANT  C
C                  TO BE RETURNED BY THE PROP2 ROUTINE                 C
C IERR    I    O   ERROR FLAG                                          C
C                  0 = SUCCESS                                         C
C                  1 = TEMPERATURE OUT OF RANGE                        C
C                  2 = UNABLE TO BRACKET (PROBABLY BECAUSE VERY CLOSE  C
C                      TO TC) RETURN VALUES FROM SATURATION EQUATIONS  C
C                  3 = UNABLE TO CONVERGE BOUNDED SOLUTION.  RETURN    C
C                      LAST GUESS FOR SOLUTION                         C
C                  4 = TEMPERATURE BELOW TRIPLE POINT BUT IN RANGE     C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 06SEP95 - INITIAL CREATION BY AHH (FROM PCOEX)                       C                                    C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 23MAY96 - AHH: RETURN BETTER NEAR-CRITICAL DENSITIES FOR IERR=2      C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 25SEP96 - AHH: SET RHOL AND RHOV FOR IERR=1 TO AVOID UNSET VARS      C
C 09JAN97 - AHH: ALLOW CALCULATIONS BELOW TRIPLE POINT                 C
C 21MAY97 - AHH: USE COMMONS FOR WATER PARAMETERS                      C
C 10NOV99 - AHH: GET BETTER INITIAL GUESSES IF NEAR CRITICAL           C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      INCLUDE 'wconst.cmn'
      INCLUDE 'wlimit.cmn'
      PARAMETER (TOL=1.D-8, TOL2=1.D-7)
      PARAMETER (PTRIP = PTRIPW/PCW, PMIN = PSMIN/PCW)
      LOGICAL LITER1
      DIMENSION PROPR(NPROP)
      DIMENSION IWANT(NPROP)
      LITER1 = .TRUE.
C PRESSURE RANGE CHECK
      IF (PR .LT. PMIN) THEN
        IERR = 1
        TR = 1.D-8
        RHOL = 0.D0
        RHOV = 0.D0
        RETURN
      ELSE IF (PR .GE. 1.D0) THEN
        IERR = 1
        TR = 1.D0
        RHOL = 1.D0
        RHOV = 1.D0
        RETURN
      ENDIF
C
C GET INITIAL GUESSES FOR PSAT, RHOL, RHOV
C
      CALL TVAP(PR, T0)
      IF (T0 .GE. .999998D0) THEN
        CALL TVAPC(PR, T0)
        RHOL0 = DLSATC(T0)
        RHOV0 = DVSATC(T0)
      ELSE
        RHOL0 = DLSAT(T0)
        RHOV0 = DVSAT(T0)
      ENDIF
C
C FIND LIQUID AND VAPOR DENSITIES AT INITIALLY GUESSED TEMPERATURE
C
      CALL DFIND(RHOL,PR,RHOL0,T0,DPD,IWANT,PROPR,IER1)
      CALL DFIND(RHOV,PR,RHOV0,T0,DPD,IWANT,PROPR,IER2)
C
C IF YOU COULDN'T FIND DENSITY ROOTS AT THE INITIAL GUESS, YOU ARE
C PROBABLY TOO CLOSE TO TC.  RETURN INIT. GUESS VALUES
C
      IF ((IER1.NE.0) .OR. (IER2.NE.0)) GO TO 666
C
C GET THE K-FACTOR AT THIS TEMP, RHOL, AND RHOV
C
      CALL KFACT(T0,RHOL,RHOV,FLIQ,FVAP,XK,IWANT,PROPR)
C
C OBJECTIVE FUNCTION IS LN OF THE K-FACTOR
C
      F0 = LOG(XK)
C
C NOW SEARCH FOR A TEMPERATURE ON THE OTHER SIDE OF EQUILIBRIUM
C IF F0 IS NEGATIVE, THE TEMPERATURE NEEDS TO BE HIGHER
C IF YOU JUMP PAST THE 2-PHASE PART OF THE ISOTHERM, HALVE THE STEP
C
      TOLD = T0
      IF (F0 .LT. 0.D0) THEN
        TINCR = .01*(1.D0-T0)
        DO 100 I=1,100
  110     TNEW = TOLD + TINCR
          CALL DFIND(RHOL,PR,RHOL0,TNEW,DPD,IWANT,PROPR,IER1)
          CALL DFIND(RHOV,PR,RHOV0,TNEW,DPD,IWANT,PROPR,IER2)
          IF ((IER1.NE.0) .OR. (IER2.NE.0)) THEN
            TINCR = 0.5*TINCR
            GO TO 110
          ENDIF
          CALL KFACT(TNEW,RHOL,RHOV,FLIQ,FVAP,XK,IWANT,PROPR)
          F1 = LOG(XK)
          IF (F1 .GT. 0.D0) THEN
            TLOW = TOLD
            THIGH = TNEW      
            FLOW = F0
            FHIGH = F1
            GO TO 201
          ELSE
            TOLD = TNEW
            F0 = F1
          ENDIF
  100   CONTINUE
C
C - IF YOU GET HERE, YOU WERE UNABLE TO BRACKET THE TEMPERATURE
C
        GO TO 666
C
      ELSE
C
C IF F WAS TOO HIGH, NEED TO LOWER THE TEMPERATURE
C
        TINCR = .01*T0
        DO 200 I=1,90
  210     TNEW = TOLD - TINCR
          CALL DFIND(RHOL,PR,RHOL0,TNEW,DPD,IWANT,PROPR,IER1)
          CALL DFIND(RHOV,PR,RHOV0,TNEW,DPD,IWANT,PROPR,IER2)
          IF ((IER1.NE.0) .OR. (IER2.NE.0)) THEN
            TINCR = 0.5*TINCR
            GO TO 210
          ENDIF
          CALL KFACT(TNEW,RHOL,RHOV,FLIQ,FVAP,XK,IWANT,PROPR)
          F1 = LOG(XK)
          IF (F1 .LT. 0.D0) THEN
            TLOW = TNEW
            THIGH = TOLD      
            FLOW = F1
            FHIGH = F0
            GO TO 201
          ELSE
            TOLD = TNEW
            F0 = F1
          ENDIF
  200   CONTINUE
C
C - IF YOU GET HERE, YOU WERE UNABLE TO BRACKET THE TEMPERATURE
C
        GO TO 666
C
      ENDIF
C
  201 CONTINUE
C
C NOW THE SATURATION TEMPERATURE IS BRACKETED BY TLOW AND THIGH, WITH
C CORRESPONDING OBJECTIVE FUNCTION VALUES OF FLOW AND FHIGH.
C CONVERGE ON THE SOLUTION BY SECANT METHOD ON 1/TR
C
      TAULOW = 1.D0/TLOW
      TAUHI  = 1.D0/THIGH
      DO 300 I=1,100
        FRDIST = -FLOW / (FHIGH-FLOW)
        IF (FRDIST .LT. 0.01) THEN
          LITER1 = .TRUE.
          FRDIST = 0.01
        ELSE IF (FRDIST .GT. 0.99) THEN
          LITER1 = .TRUE.
          FRDIST = 0.99
        ENDIF
        TAUNEW = TAULOW + FRDIST*(TAUHI-TAULOW)
        TNEW = 1.D0/TAUNEW
        CALL DFIND(RHOL,PR,RHOL0,TNEW,DPD,IWANT,PROPR,IER1)
        CALL DFIND(RHOV,PR,RHOV0,TNEW,DPD,IWANT,PROPR,IER2)
        CALL KFACT(TNEW,RHOL,RHOV,FLIQ,FVAP,XK,IWANT,PROPR)
        FNEW = LOG(XK)
C
C CHECK FOR CONVERGENCE
C BUT NOT IF IT IS THE FIRST ITERATION OR YOU BOUNDED THE STEP
C
        IF (LITER1) THEN
          LITER1 = .FALSE.
        ELSE
          IF (DABS(FNEW) .LT. TOL) GO TO 777
          IF ((PR .LT. PTRIP) .AND. (DABS(FNEW) .LT. TOL2)) GO TO 777
        ENDIF
C
C MAKE NEW BOUNDS DEPENDING ON WHETHER NEW GUESS WAS HIGH OR LOW
C
        IF (FNEW .GT. 0.D0) THEN
          FHIGH = FNEW
          THIGH = TNEW
          TAUHI = TAUNEW
        ELSE
          FLOW = FNEW
          TLOW = TNEW
          TAULOW = TAUNEW
        ENDIF
  300 CONTINUE
C
C IF YOU GET TO HERE, YOU DID NOT CONVERGE ON THE BOUNDED ROOT
C
      IERR = 3
      TR = TNEW
      RETURN
C
  666 CONTINUE
C
C GET HERE FOR INITIAL FAILURE OR FAILURE TO BRACKET SOLUTION
C RETURN INITIAL GUESSES
C
      IERR = 2
      TR = T0
      RHOL = RHOL0
      RHOV = RHOV0
      IF (TR .GT. 0.99999) THEN
        RHOL = DLSATC(TR)
        RHOV = DVSATC(TR)
      ENDIF
      RETURN
C
  777 CONTINUE
C
C CONVERGENCE
C
      IERR = 0
C
C NOTE EXTRAPOLATION BELOW TRIPLE POINT
C
      IF (PR .LT. PTRIP) IERR = 4
C  
      TR = TNEW
      RETURN
      END
