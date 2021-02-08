C
      SUBROUTINE TMELT(TK, PMPA, IFORM, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** TMELT ***                                    C
C THIS ROUTINE GIVES THE MELTING TEMPERATURE TK AS A FUNCTION OF       C
C PRESSURE PMPA USING THE EQUATIONS OF WAGNER ET AL.  (J. PHYS.        C
C CHEM. REF. DATA, 23, 515 (1994).  IT SOLVES BY ITERATIVELY CALLING   C
C PMELT1 (FOR ICE I) OR PMELT2 (FOR HIGH-PRESSURE FORMS).              C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    O   MELTING TEMPERATURE IN K                            C
C PMPA    R    I   PRESSURE IN MPA                                     C
C IFORM   I    O   NUMBER OF ICE FORM INVOLVED IN THE EQUILIBRIUM      C
C IERR    I    O   RETURN CODE:                                        C
C                  0 = SUCCESS                                         C
C                  1 = PRESSURE BELOW VALID RANGE (611.657D-6 MPA)     C
C                  2 = PRESSURE ABOVE VALID RANGE (2216 MPA)           C
C                  IF 1 OR 2, TMIN OR TMAX IS RETURNED                 C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 29SEP95 - INITIAL CREATION BY AHH                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (PMAX = 20618.D0, PMIN = 611.657D-6)
      PARAMETER (TTOP = 715.D0, TBOT = 273.16D0)
      PARAMETER (T13 = 251.165D0, P13 = 209.9D0)
      PARAMETER (TOL = 1.D-8)
      LOGICAL LHIGHP
C
C FIRST, ESTABLISH BOUNDS AND SEE IF YOU ARE INSIDE
C
      IF (PMPA .LT. PMIN) THEN
        TK = TBOT
        IERR = 1
      ELSE IF (PMPA .GT. PMAX) THEN
        TK = TTOP
        IERR = 2
      ELSE
C
C ESTABLISH WHETHER YOU ARE ON THE ICE-I BRANCH OR THE HIGH-P PART
C
        IF (PMPA .GT. P13) THEN
          LHIGHP = .TRUE.
          THIGH = TTOP
          TLOW = T13
          CALL PMELT2(TTOP, PTOP, IDUM, IERR)
          FHIGH = LOG(PTOP/PMPA)
          FLOW = LOG(P13/PMPA)
        ELSE
          LHIGHP = .FALSE.
          IFORM = 1
          THIGH = TBOT
          TLOW = T13
          CALL PMELT1(TLOW, PLOW, IERR)
          FHIGH = -LOG(PMIN/PMPA)
          FLOW = -LOG(PLOW/PMPA)
        ENDIF
C
C SOLUTION IS BOUNDED BETWEEN TMAX AND TMIN, SO DO BOUNDED SECANT
C WITH LOG P 
C
        DO 300 I=1,100
          FRDIST = -FLOW / (FHIGH-FLOW)
          IF (FRDIST .LT. 0.01) FRDIST = 0.01
          IF (FRDIST .GT. 0.99) FRDIST = 0.99
          TNEW = TLOW + FRDIST*(THIGH-TLOW)
          IF (LHIGHP) THEN
            CALL PMELT2(TNEW, PNEW, IFORM, IERR)
            FNEW = LOG(PNEW/PMPA)
          ELSE
            CALL PMELT1(TNEW, PNEW, IERR)
            FNEW = -LOG(PNEW/PMPA)
          ENDIF
C
C CHECK FOR CONVERGENCE
C
          IF (DABS(FNEW) .LT. TOL) GO TO 777
C
C MAKE NEW BOUNDS DEPENDING ON WHETHER NEW GUESS WAS HIGH OR LOW
C
          IF (FNEW .GT. 0.D0) THEN
            FHIGH = FNEW
            THIGH = TNEW
          ELSE
            FLOW = FNEW
            TLOW = TNEW
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
