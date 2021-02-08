C
      SUBROUTINE PMELT2(TK, PMPA, IFORM, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PMELT2 ***                                   C
C THIS ROUTINE GIVES THE MELTING PRESSURE PMPA AS A FUNCTION OF        C
C TEMPERATURE TK USING THE EQUATION OF WAGNER ET AL.  (J. PHYS.        C
C CHEM. REF. DATA, 23, 515 (1994).  FOR COEXISTENCE WITH HIGHER-P      C
C (NON-I) FORMS OF ICE                                                 C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    I   TEMPERATURE IN K                                    C
C PMPA    R    O   MELTING PRESSURE IN MPA                             C
C IFORM   I    O   FORM OF ICE FOR WHICH EQUILIBRIUM EXISTS            C
C IERR    I    O   RETURN CODE:                                        C
C                  0 = SUCCESS                                         C
C                  1 = TEMPERATURE BELOW VALID RANGE (251.165 K)       C
C                  2 = TEMPERATURE ABOVE VALID RANGE (715. K)          C
C                  IF 1 OR 2, PMPA IS RETURNED AS ZERO                 C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 28SEP95 - INITIAL CREATION BY AHH                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (TMIN3 = 251.165D0, PN3 = 209.9D0)
      PARAMETER (TMIN5 = 256.164D0, PN5 = 350.1D0)
      PARAMETER (TMIN6 = 273.31D0,  PN6 = 632.4D0)
      PARAMETER (TMIN7 = 355.D0,    PN7 = 2216.D0, TMAX7 = 715.D0)
      PMPA = 0.D0
      IERR = 0
      IFORM = 0
      IF (TK .LT. TMIN3) THEN
        IERR = 1
      ELSE IF (TK .GT. TMAX7) THEN
        IERR = 2
      ELSE IF (TK .LE. TMIN5) THEN
        THETA = TK / TMIN3
        PRAT = 1.D0 - .295252*(1.D0-THETA**60)
        PMPA = PRAT * PN3
        IFORM = 3
      ELSE IF (TK .LE. TMIN6) THEN
        THETA = TK / TMIN5
        PRAT = 1.D0 - 1.18721*(1.D0-THETA**8)
        PMPA = PRAT * PN5
        IFORM = 5
      ELSE IF (TK .LE. TMIN7) THEN
        THETA = TK / TMIN6
        PRAT = 1.D0 - 1.07476*(1.D0-THETA**4.6)
        PMPA = PRAT * PN6
        IFORM = 6
      ELSE
        THETA = TK / TMIN7
        TERM1 = 0.173683D1 * (1.D0-1.D0/THETA)
        TERM2 = -0.544606D-1 * (1.D0-THETA**5)
        TERM3 = 0.806106D-7 * (1.D0-THETA**22)
        PRAT = DEXP(TERM1+TERM2+TERM3)
        PMPA = PRAT * PN7
        IFORM = 7
      ENDIF
      RETURN
      END
