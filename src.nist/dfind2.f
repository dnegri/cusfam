C
      SUBROUTINE DFIND2(DOUT, P, D, TR, DPD, IWANT, PROPR, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** DFIND2 ***                                   C
C THIS ROUTINE FINDS THE REDUCED DENSITY CORRESPONDING TO A GIVEN      C
C REDUCED TEMPERATURE TR AND REDUCED (P/PC) PRESSURE P.  AN            C
C INITIAL GUESS (D) IS REQUIRED, AND THE VALUE OF DPDD IS ALSO         C
C RETURNED.  THIS ROUTINE IS SPECIFICALLY FOR A PURE FLUID AT TR < 1   C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C DOUT    R    O   REDUCED DENSITY RHO/RHOC AT TR AND P                C
C P       R    I   REDUCED (P/PC) PRESSURE                             C
C D       R    I   INITIAL GUESS FOR REDUCED DENSITY, RHO/RHOC         C
C TR      R    I   REDUCED TEMPERATURE, T/TC                           C
C DPD     R    O   FIRST DERIVATIVE OF P WITH RESPECT TO D AT DOUT     C
C IWANT   IA   -   PROPERTY REQUEST VECTOR FOR USE IN PROPS2 ROUTINE   C
C PROPR   RA   O   VECTOR OF REDUCED PROPERTIES AS REQUESTED BY IWANT  C
C                  TO BE RETURNED BY THE PROP2 ROUTINE                 C
C IERR    I    O   RETURN STATUS FLAG:                                 C
C                  0: SUCCESSFUL CONVERGENCE                           C
C                  1: NO ROOT FOR REQUESTED PHASE.  RETURNS D WHERE    C
C                     DPD = 0                                          C
C                  2: UNABLE TO BOUND ROOT                             C
C                  3: UNABLE TO CONVERGE BOUNDED ROOT                  C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 22AUG95 - INITIAL CREATION BY AHH                                    C
C 30AUG95 - AHH: ADD IERR FLAG, HANDLE CASE WHERE REQUESTED V OR L     C
C                ROOT DOES NOT EXIST                                   C
C 31AUG95 - TIGHTEN CONVERGENCE TOLERANCE WHEN NEAR TC                 C
C 01SEP95 - FORCE AT LEAST A 1% STEP IN BOUNDED SECANT SEARCH          C
C 06SEP95 - AHH: BOUND INCREASE WHEN LOOKING FOR VAPOR ROOT            C
C 07SEP95 - AHH: FIX SEARCH FOR POSITIVE DP FOR VAPOR ROOTS            C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 04OCT95 - AHH: IMPROVE FOR LOW DENSITIES                             C
C 19OCT95 - AHH: IMPROVE CONVERGENCE ONCE BOUNDED                      C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 09JUN97 - AHH: LOOSEN TOLERANCE FOR VERY LOW TEMPERATURES            C
C 10NOV99 - AHH: CHANGES TO CONVERGE BETTER NEAR CRITICAL POINT        C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWANT(NPROP)
      LOGICAL LITER1
      LITER1 = .TRUE.
      TOL = 1.0D-8
      TOLMIN = 1.D-11
      IF ((TR .GT. 0.9999D0) .AND. (P .GT. 0.98D0) .AND. 
     1    (P .LT. 1.02D0)) THEN
        TOL = 1.D-9
        TOLMIN = 1.D-13
      ENDIF
      IF (TR .LT. 0.7D0) TOL = 1.0D-7
      IF (TR .LT. 0.422D0) TOL = 1.0D-6
C
C FIRST, EVALUATE P AND DP AT INITIAL GUESS (IF GUESS IS OK)
C
      DD = D
      IF(DD.LT.1.D-20) DD=1.D-20
      IF(DD.GT.5.5) DD=5.5         
      CALL PDP(DD,TR,PNEW,DPNEW,IWANT,PROPR)
C
C FROM INITIAL GUESS, IDENTIFY WHETHER A VAPOR OR LIQUID ROOT IS WANTED
C
      IF (DD .GT. 1.D0) THEN
C
C LIQUID ROOT DESIRED.  IF YOU HAVE FOUND DP > 0, SEARCH TO FIND ROOT
C IF DP < 0, MUST GO UP UNTIL A POSITIVE DP IS FOUND
C AT THAT POINT, TREAT LIKE "NORMAL" CASE AND GO TO SEARCH FOR BOUND
C        
        IF (DPNEW .LT. 0.D0) THEN
          DOLD = DD
          DO 50 I=1,100
            DNEW = DOLD*1.05
            CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
            DOLD = DNEW
            DPOLD = DPNEW
            IF (DPNEW .GT. 0.D0) GO TO 99
   50     CONTINUE
        ENDIF
C
        DOLD = DD
        POLD = PNEW
        DPOLD = DPNEW
C
C IF P TOO HIGH, GO DOWN IN DENSITY UNTIL SOLUTION BOUNDED
C
   99   CONTINUE
        IF (PNEW .GT. P) THEN
          DO 100 I=1,100
            DNEW = DOLD - 0.1*(DOLD-1.)
            CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
            IF (DPNEW .LT. 0.D0) THEN
C
C IF NEGATIVE DPD ENCOUNTERED, SEARCH FOR PLACE WHERE DPD=0.  IF P THERE
C IS STILL TOO HIGH, THEN THERE IS NO LIQUID ROOT.  BUT IF YOU GET A 
C POINT WITH DPD>0 AND P TOO LOW, THE ROOT IS BOUNDED AND YOU CAN SEARCH.
C
              FLOW = DPNEW
              FHIGH = DPOLD
              DLOW = DNEW
              DHIGH = DOLD
              DPLOW = DPNEW
              DPHIGH = DPOLD
              PSAV = POLD
              DSAV = DOLD
              DPSAV = DPOLD
              DO 110 J=1,100
                FRDIST = -FLOW / (FHIGH-FLOW)
                DNEW = DLOW + FRDIST*(DHIGH-DLOW)
                CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
C
C CHECK FOR BOUNDING OF DENSITY ROOT
C
                IF ((DPNEW.GT.0.D0) .AND. (PNEW.LE.P)) THEN
                  PHIGH = PSAV
                  DHIGH = DSAV
                  DPHIGH = DPSAV
                  PLOW = PNEW
                  DLOW = DNEW
                  DPLOW = DPNEW
                  GO TO 444
                ENDIF
C
C CHECK FOR CONVERGENCE ON DP=0
C
                IF(DABS(DPNEW).LT.TOL) THEN
                  DOUT = DNEW
                  DPD = DPNEW
                  IERR = 1
                  RETURN
                ENDIF
C
C MAKE NEW BOUNDS DEPENDING ON WHETHER NEW GUESS WAS HIGH OR LOW
C
                FNEW = DPNEW
                IF (FNEW .GE. 0.D0) THEN
                  FHIGH = FNEW
                  DHIGH = DNEW
                  DPHIGH = DPNEW
                ELSE
                  FLOW = FNEW
                  DLOW = DNEW
                  DPLOW = DPNEW
                ENDIF
  110         CONTINUE
            ENDIF
            IF (PNEW .LE. P) THEN
              PHIGH = POLD
              DHIGH = DOLD
              DPHIGH = DPOLD
              PLOW = PNEW
              DLOW = DNEW
              DPLOW = DPNEW
              GO TO 444
            ENDIF
            DOLD = DNEW
            POLD = PNEW
            DPOLD = DPNEW
  100     CONTINUE
C
C IF YOU STILL HAVEN'T BRACKETED IT, GIVE UP
C
          DOUT = DNEW
          DPD = DPNEW
          IERR = 2
          RETURN
        ELSE
C
C IF P TOO LOW, GO UP IN DENSITY UNTIL SOLUTION BOUNDED
C
          DO 200 I=1,100
            DNEW = DOLD*1.1
            CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
            IF (PNEW .GE. P) THEN
              PLOW = POLD
              DLOW = DOLD
              DPLOW = DPOLD
              PHIGH = PNEW
              DHIGH = DNEW
              DPHIGH = DPNEW
              GO TO 444
            ENDIF
            DOLD = DNEW
            POLD = PNEW
            DPOLD = DPNEW
  200     CONTINUE
C
C IF YOU STILL HAVEN'T BRACKETED IT, GIVE UP
C
          DOUT = DNEW
          DPD = DPNEW
          IERR = 2
          RETURN
C
        ENDIF
      ELSE
C
C VAPOR ROOT DESIRED.  PROCEDURE IS MIRROR IMAGE OF LIQUID PROCEDURE
C IF YOU HAVE FOUND DP > 0, SEARCH TO FIND ROOT
C IF DP < 0, MUST GO DOWN UNTIL A NEGATIVE DP IS FOUND
C AT THAT POINT, TREAT LIKE "NORMAL" CASE AND GO TO SEARCH FOR BOUND
C        
        IF (DPNEW .LT. 0.D0) THEN
          DOLD = DD
          DO 250 I=1,100
            DNEW = DOLD*0.95
            CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
            DOLD = DNEW
            DPOLD = DPNEW
            IF (DPNEW .GT. 0.D0) GO TO 299
  250     CONTINUE
        ENDIF
C
        DOLD = DD
        POLD = PNEW
        DPOLD = DPNEW
C
C IF P TOO LOW, GO UP IN DENSITY UNTIL SOLUTION BOUNDED
C
  299   CONTINUE
        IF (PNEW .LT. P) THEN
          DO 300 I=1,1000
            DINCR = MIN(1.D-2*(1.D0-DOLD), DOLD*1.1)
            DNEW = DOLD + DINCR
            CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
            IF (DPNEW .LT. 0.D0) THEN
C
C IF NEGATIVE DPD ENCOUNTERED, SEARCH FOR PLACE WHERE DPD=0.  IF P THERE
C IS STILL TOO LOW, THEN THERE IS NO VAPOR ROOT.  BUT IF YOU GET A 
C POINT WITH DPD>0 AND P TOO HIGH, THE ROOT IS BOUNDED AND YOU CAN SEARCH.
C
              FLOW = DPNEW
              FHIGH = DPOLD
              DLOW = DNEW
              DHIGH = DOLD
              DPLOW = DPNEW
              DPHIGH = DPOLD
              PSAV = POLD
              DSAV = DOLD
              DPSAV = DPOLD
              DO 210 J=1,100
                FRDIST = -FLOW / (FHIGH-FLOW)
                DNEW = DLOW + FRDIST*(DHIGH-DLOW)
                CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
C
C CHECK FOR BOUNDING OF DENSITY ROOT
C
                IF ((DPNEW.GT.0.D0) .AND. (PNEW.GE.P)) THEN
                  PLOW = PSAV
                  DLOW = DSAV
                  DPLOW = DPSAV
                  PHIGH = PNEW
                  DHIGH = DNEW
                  DPHIGH = DPNEW
                  GO TO 444
                ENDIF
C
C CHECK FOR CONVERGENCE ON DP=0
C
                IF(DABS(DPNEW).LT.TOL) THEN
                  DOUT = DNEW
                  DPD = DPNEW
                  IERR = 1
                  RETURN
                ENDIF
C
C MAKE NEW BOUNDS DEPENDING ON WHETHER NEW GUESS WAS HIGH OR LOW
C
                FNEW = DPNEW
                IF (FNEW .GE. 0.D0) THEN
                  FHIGH = FNEW
                  DHIGH = DNEW
                  DPHIGH = DPNEW
                ELSE
                  FLOW = FNEW
                  DLOW = DNEW
                  DPLOW = DPNEW
                ENDIF
  210         CONTINUE
            ENDIF
C
            IF (PNEW .GE. P) THEN
              PLOW = POLD
              DLOW = DOLD
              DPLOW = DPOLD
              PHIGH = PNEW
              DHIGH = DNEW
              DPHIGH = DPNEW
              GO TO 444
            ENDIF
            DOLD = DNEW
            POLD = PNEW
            DPOLD = DPNEW
  300     CONTINUE
C
C IF YOU STILL HAVEN'T BRACKETED IT, GIVE UP
C
          DOUT = DNEW
          DPD = DPNEW
          IERR = 2
          RETURN
        ELSE
C
C IF P TOO HIGH, GO DOWN IN DENSITY UNTIL SOLUTION BOUNDED
C
          DO 400 I=1,100
            DNEW = DOLD*0.9
            CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
            IF (PNEW .LE. P) THEN
              PHIGH = POLD
              DHIGH = DOLD
              DPHIGH = DPOLD
              PLOW = PNEW
              DLOW = DNEW
              DPLOW = DPNEW
              GO TO 444
            ENDIF
            DOLD = DNEW
            POLD = PNEW
            DPOLD = DPNEW
  400     CONTINUE
C
C IF YOU STILL HAVEN'T BRACKETED IT, GIVE UP
C
          DOUT = DNEW
          DPD = DPNEW
          IERR = 2
          RETURN
C
        ENDIF
      ENDIF
  444 CONTINUE
C
C SOLUTION IS NOW BRACKETED
C SOLVE BY NEWTON BASED ON DERIVATIVE AT NEAREST END, BUT DON'T
C LET IT CROSS THE BOUNDARY
C MAKE IT ITERATE AT LEAST ONCE, AND DON'T LET IT CONVERGE IF YOU
C HAD TO BOUND THE STEP
C
      FLOW = PLOW - P
      FHIGH = PHIGH - P
      DO 500 I=1,200
        IF (DABS(FLOW) .LT. DABS(FHIGH)) THEN
          DNEW = DLOW + DABS(FLOW/DPLOW)
          DBOUND = DLOW + 0.9*(DHIGH-DLOW)
          IF (DNEW .GT. DBOUND) THEN
            DNEW = DBOUND
            LITER1 = .TRUE.
          ENDIF
        ELSE
          DNEW = DHIGH - DABS(FHIGH/DPHIGH)
          DBOUND = DLOW + 0.1*(DHIGH-DLOW)
          IF (DNEW .LT. DBOUND) THEN
            DNEW = DBOUND
            LITER1 = .TRUE.
          ENDIF
        ENDIF
        CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
C
C CHECK FOR CONVERGENCE
C
        IF (LITER1) THEN
          LITER1 = .FALSE.
        ELSE
          TOL2 = MIN(TOL,TOL*DPNEW)
          TOL2 = MAX(TOLMIN,TOL2)
          IF(DABS(1.D0-PNEW/P).LT.TOL2) GO TO 999
        ENDIF
C
C MAKE NEW BOUNDS DEPENDING ON WHETHER NEW GUESS WAS HIGH OR LOW
C
        FNEW = PNEW - P
        IF (FNEW .GE. 0.D0) THEN
          FHIGH = FNEW
          DHIGH = DNEW
          DPHIGH = DPNEW
        ELSE
          FLOW = FNEW
          DLOW = DNEW
          DPLOW = DPNEW
        ENDIF
  500 CONTINUE
C
C FAILURE TO CONVERGE
C
      DOUT = DNEW
      DPD = DPNEW
      IERR = 3
      RETURN
C
  999 CONTINUE
C
C CONVERGENCE
C
      IERR = 0
      DOUT = DNEW
      DPD = DPNEW
      RETURN       
      END
