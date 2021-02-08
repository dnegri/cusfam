      SUBROUTINE PMELT(TK, NROOTS, PMPA1, PMPA2, IFORM, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PMELT ***                                    C
C THIS ROUTINE GIVES THE MELTING PRESSURE(S) PMPA AS A FUNCTION OF     C
C TEMPERATURE TK USING THE EQUATIONS OF WAGNER ET AL.  (J. PHYS.       C
C CHEM. REF. DATA, 23, 515 (1994).  ALL FORMS OF ICE ARE CONSIDERED,   C
C SO IT IS POSSIBLE TO HAVE TWO ROOTS.                                 C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    I   TEMPERATURE IN K                                    C
C NROOTS  I    O   NUMBER OF PRESSURES AT WHICH MELTING HAPPENS AT TK  C
C PMPA1   R    O   MELTING PRESSURE IN MPA FOR FIRST ROOT, IF ANY      C
C PMPA2   R    O   MELTING PRESSURE IN MPA FOR SECOND ROOT, IF ANY     C
C                  IF 2 ROOTS, PMPA1 HAS LOWEST PRESSURE (ICE I)       C 
C IFORM   I    O   FORM OF ICE WITH WHICH FLUID COEXISTS.  IF NROOTS   C
C                  IS 2, THIS CONTAINS THE HIGH-PRESSURE FORM SINCE    C
C                  ROOT 1 MUST BE ICE I.                               C
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
      PARAMETER (TMAX = 715.D0, TMIN = 251.165D0, T1MAX = 273.16D0)
      PMPA1 = 0.D0
      PMPA2 = 0.D0
      IERR = 0
      IFORM = 0
      NROOTS = 0
      IF (TK .LT. TMIN) THEN
        IERR = 1
      ELSE IF (TK .GT. TMAX) THEN
        IERR = 2
      ELSE IF (TK .LE. T1MAX) THEN
C
C BOTH AN ICE-I ROOT AND A ROOT FOR A HIGHER-P FORM
C
        CALL PMELT1(TK, PMPA1, IDUM)
        CALL PMELT2(TK, PMPA2, IFORM, IDUM)
        NROOTS = 2
      ELSE
C
C ONLY ROOT IS A HIGHER-PRESSURE FORM OF ICE
C
        CALL PMELT2(TK, PMPA1, IFORM, IDUM)
        NROOTS = 1
      ENDIF
      RETURN
      END
