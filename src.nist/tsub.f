C
      SUBROUTINE TSUB(TK, PMPA, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** TSUB ***                                     C
C THIS ROUTINE GIVES THE SUBLIMATION TEMPERATURE TK AS A FUNCTION OF   C
C PRESSURE PMPA USING THE EQUATION OF WAGNER ET AL.  (J. PHYS.         C
C CHEM. REF. DATA, 23, 515 (1994).  IT SOLVES BY ITERATIVELY CALLING   C
C PSUB.                                                                C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    O   SUBLIMATION TEMPERATURE IN K                        C
C PMPA    R    I   PRESSURE IN MPA                                     C
C IERR    R    O   RETURN CODE:                                        C
C                  0 = SUCCESS                                         C
C                  1 = PRESSURE BELOW VALID RANGE (TK=190 K)           C
C                  2 = PRESSURE ABOVE VALID RANGE (TK=273.16 K)        C
C                  IF 1 OR 2, 190 OR 273.16 IS RETURNED                C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 27SEP95 - INITIAL CREATION BY AHH                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (TMAX = 273.16D0, TMIN = 190.D0)
      PARAMETER (TOL = 1.D-8)
C
C FIRST, ESTABLISH BOUNDS AND SEE IF YOU ARE INSIDE
C
      CALL PSUB (TMIN, PMIN, IERR)
      CALL PSUB (TMAX, PMAX, IERR)
      IF (PMPA .LT. PMIN) THEN
        TK = TMIN
        IERR = 1
      ELSE IF (PMPA .GT. PMAX) THEN
        TK = TMAX
        IERR = 2
      ELSE
C
C SOLUTION IS BOUNDED BETWEEN TMAX AND TMIN, SO DO BOUNDED SECANT
C WITH LOG P LINEAR IN 1/T
C
        FHIGH = LOG(PMAX/PMPA)
        FLOW = LOG(PMIN/PMPA)
        TAULOW = 1.D0/TMIN
        TAUHI  = 1.D0/TMAX
        DO 300 I=1,100
          FRDIST = -FLOW / (FHIGH-FLOW)
          IF (FRDIST .LT. 0.01) FRDIST = 0.01
          IF (FRDIST .GT. 0.99) FRDIST = 0.99
          TAUNEW = TAULOW + FRDIST*(TAUHI-TAULOW)
          TNEW = 1.D0/TAUNEW
          CALL PSUB(TNEW, PNEW,  IERR)
          FNEW = LOG(PNEW/PMPA)
C
C CHECK FOR CONVERGENCE
C
          IF (DABS(FNEW) .LT. TOL) GO TO 777
C
C MAKE NEW BOUNDS DEPENDING ON WHETHER NEW GUESS WAS HIGH OR LOW
C
          IF (FNEW .GT. 0.D0) THEN
            FHIGH = FNEW
            TAUHI = TAUNEW
          ELSE
            FLOW = FNEW
            TAULOW = TAUNEW
          ENDIF
  300   CONTINUE
C
C IF YOU GET TO HERE, YOU DID NOT CONVERGE ON THE BOUNDED ROOT
C
        IERR = 3
        TK = TNEW
        RETURN
C
C CONVERGENCE
C
  777   CONTINUE
        IERR = 0
        TK = TNEW
      ENDIF
      RETURN
      END
