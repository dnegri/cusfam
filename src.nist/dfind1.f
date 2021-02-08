C
      SUBROUTINE DFIND1(DOUT, P, D, TR, DPD, IWANT, PROPR, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** DFIND1 ***                                   C
C THIS ROUTINE FINDS THE REDUCED DENSITY CORRESPONDING TO A GIVEN      C
C REDUCED TEMPERATURE TR AND REDUCED (P/PC) PRESSURE P.  AN            C
C INITIAL GUESS (D) IS REQUIRED, AND THE VALUE OF DPDD IS ALSO         C
C RETURNED.  THIS ROUTINE IS SPECIFICALLY FOR A PURE FLUID AT TR > 1   C
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
C IERR    I    O   RETURN STATUS CODE                                  C
C                  0: CONVERGED                                        C
C                 -2: UNABLE TO BOUND ROOT                             C
C                 -3: UNABLE TO CONVERGE BOUNDED ROOT                  C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 21AUG95 - INITIAL CREATION BY AHH                                    C
C 22AUG95 - AHH: SPEED UP BOUNDARY-FINDING A LITTLE                    C
C 30AUG95 - AHH: ADD IERR FLAG TO RETURN STATUS                        C
C 19SEP95 - AHH: FORCE AT LEAST A 1% STEP IN BOUNDED SECANT SEARCH     C
C 19SEP95 - AHH: SET DOLD AND POLD FOR CASE JUST ABOVE TC              C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWANT(NPROP)
      PARAMETER (TCLOSE=1.1)
      PARAMETER (TOL = 1.0D-8)
C
C FIRST, EVALUATE P AND DP AT INITIAL GUESS (IF GUESS IS OK)
C
      DD = D
      IF(DD.LE.1.D-8) DD=1.D-8
      IF(DD.GT.5.5) DD=5.5         
      CALL PDP(DD,TR,PNEW,DPD,IWANT,PROPR)
C
C IF NOT TOO CLOSE TO TC, DO NEWTON ITERATION
C
      IF (TR .GT. TCLOSE) THEN
        DOLD = DD
        ERROLD = PNEW - P
   10   STEP = -ERROLD / DPD
        IF (DABS(STEP) .GT. 0.5*DOLD) THEN
          STEP = SIGN(0.5*DOLD, STEP)
        ENDIF
        DNEW = DOLD + STEP
   20   CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
C
C CHECK FOR CONVERGENCE
C
        IF(DABS(1.D0-PNEW/P).LT.TOL) GO TO 999
C
C IF NOT CONVERGED, SEE IF ERROR HAS DECREASED.  IF NOT, CUT STEP IN HALF
C IF IT HAS, TAKE THE NEXT NEWTON STEP
C
        ERRNEW = PNEW - P
        IF (DABS(ERRNEW) .GT. DABS(ERROLD)) THEN
          DNEW = 0.5*(DNEW+DOLD)
          GO TO 20
        ENDIF
        ERROLD = ERRNEW
        DOLD = DNEW
        DPD = DPNEW
        GO TO 10
      ELSE
C
C IF CLOSE TO TC, WORK BY BOUNDING THE SOLUTION
C
        IF (PNEW .GT. P) THEN
C
C IF P TOO HIGH, GO DOWN IN DENSITY UNTIL SOLUTION BOUNDED
C
          DOLD = DD
          POLD = PNEW
          DO 100 I=1,9
            DNEW = (10-I)*0.1*DD
            CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
            IF (PNEW .LE. P) THEN
              PHIGH = POLD
              DHIGH = DOLD
              PLOW = PNEW
              DLOW = DNEW
              GO TO 222
            ENDIF
            DOLD = DNEW
            POLD = PNEW
  100     CONTINUE
C
C IF YOU GET HERE, TRY SUCCESSIVE HALVING OF THE DENSITY
C
          DO 110 I=1,20
            DNEW = 0.5*DOLD
            CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
            IF (PNEW .LE. P) THEN
              PHIGH = POLD
              DHIGH = DOLD
              PLOW = PNEW
              DLOW = DNEW
              GO TO 222
            ENDIF
            DOLD = DNEW
            POLD = PNEW
  110     CONTINUE
C
C IF YOU STILL HAVEN'T BRACKETED IT, GIVE UP
C
          IERR = -2
          DOUT = DNEW
          DPD = DPNEW
          RETURN
        ELSE
C
C IF P TOO LOW, GO UP IN DENSITY UNTIL SOLUTION BOUNDED
C
          DOLD = DD
          POLD = PNEW
          DO 200 I=1,100
            DNEW = (10+I)*0.1*DD
            CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
            IF (PNEW .GE. P) THEN
              PLOW = POLD
              DLOW = DOLD
              PHIGH = PNEW
              DHIGH = DNEW
              GO TO 222
            ENDIF
            DOLD = DNEW
            POLD = PNEW
  200     CONTINUE
C
C IF YOU STILL HAVEN'T BRACKETED IT, GIVE UP
C
          IERR = -2
          DOUT = DNEW
          DPD = DPNEW
          RETURN
C
        ENDIF
  222   CONTINUE
C
C SOLUTION IS NOW BRACKETED, USE BOUNDED SECANT TO SOLVE
C
        FLOW = PLOW - P
        FHIGH = PHIGH - P
        DO 300 I=1,100
          FRDIST = -FLOW / (FHIGH-FLOW)
          IF (FRDIST .LT. 0.01) FRDIST = 0.01
          IF (FRDIST .GT. 0.99) FRDIST = 0.99
          DNEW = DLOW + FRDIST*(DHIGH-DLOW)
          CALL PDP(DNEW,TR,PNEW,DPNEW,IWANT,PROPR)
C
C CHECK FOR CONVERGENCE
C
          IF(DABS(1.D0-PNEW/P).LT.TOL) GO TO 999
C
C MAKE NEW BOUNDS DEPENDING ON WHETHER NEW GUESS WAS HIGH OR LOW
C
          FNEW = PNEW - P
          IF (FNEW .GE. 0.D0) THEN
            FHIGH = FNEW
            DHIGH = DNEW
          ELSE
            FLOW = FNEW
            DLOW = DNEW
          ENDIF
  300   CONTINUE
C
C FAILURE TO CONVERGE
C
        IERR = -3
        DOUT = DNEW
        DPD = DPNEW
        RETURN
      ENDIF
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
