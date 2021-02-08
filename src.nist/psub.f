C
      SUBROUTINE PSUB(TK, PMPA, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PSUB ***                                     C
C THIS ROUTINE GIVES THE SUBLIMATION PRESSURE PMPA AS A FUNCTION OF    C
C TEMPERATURE TK USING THE EQUATION OF WAGNER ET AL.  (J. PHYS.        C
C CHEM. REF. DATA, 23, 515 (1994)                                      C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    I   TEMPERATURE IN K                                    C
C PMPA    R    O   SUBLIMATION PRESSURE IN MPA                         C
C IERR    R    O   RETURN CODE:                                        C
C                  0 = SUCCESS                                         C
C                  1 = TEMPERATURE BELOW VALID RANGE (190 K)           C
C                  2 = TEMPERATURE ABOVE VALID RANGE (273.16 K)        C
C                  IF 1 OR 2, AN EXTRAPOLATION IS STILL RETURNED       C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 27SEP95 - INITIAL CREATION BY AHH                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (TN = 273.16D0, TMIN = 190.D0, PN = 611.657D-6)
      DATA A1, A2 /-13.928169D0, 34.7078238D0/
      IERR = 0
      IF (TK .LT. TMIN) THEN
        IERR = 1
      ELSE IF (TK .GT. TN) THEN
        IERR = 2
      ENDIF
      THETA = TK / TN
      FSUB = A1*(1.D0-THETA**(-1.5D0)) + A2*(1.D0-THETA**(-1.25D0))
      PMPA = DEXP(FSUB) * PN
      RETURN
      END
