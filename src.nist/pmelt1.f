C
      SUBROUTINE PMELT1(TK, PMPA, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PMELT1 ***                                   C
C THIS ROUTINE GIVES THE MELTING PRESSURE PMPA AS A FUNCTION OF        C
C TEMPERATURE TK USING THE EQUATION OF WAGNER ET AL.  (J. PHYS.        C
C CHEM. REF. DATA, 23, 515 (1994).  FOR COEXISTENCE WITH ICE I         C                                      C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    I   TEMPERATURE IN K                                    C
C PMPA    R    O   MELTING PRESSURE IN MPA                             C
C IERR    I    O   RETURN CODE:                                        C
C                  0 = SUCCESS                                         C
C                  1 = TEMPERATURE BELOW VALID RANGE (251.165 K)       C
C                  2 = TEMPERATURE ABOVE VALID RANGE (273.16 K)        C
C                  IF 1 OR 2, PMPA IS RETURNED AS ZERO                 C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 28SEP95 - INITIAL CREATION BY AHH                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (TN = 273.16D0, TMIN = 251.165D0, PN = 611.657D-6)
      PMPA = 0.D0
      IERR = 0
      IF (TK .LT. TMIN) THEN
        IERR = 1
      ELSE IF (TK .GT. TN) THEN
        IERR = 2
      ELSE
        THETA = TK / TN
        TERM2 = -0.626D6 * (1.D0-THETA**(-3))
        TERM3 = 0.197135D6 * (1.D0-THETA**(21.2D0))
        PRAT = 1.D0 + TERM2 + TERM3
        PMPA = PRAT * PN
      ENDIF
      RETURN
      END
