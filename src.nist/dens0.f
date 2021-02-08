C
      SUBROUTINE DENS0(DOUT, PMPA, TK, DPD, IWORK, PROPR, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** DENS0 ***                                    C
C THIS ROUTINE FINDS THE DENSITY (IN KG/M3) CORRESPONDING TO A GIVEN   C
C TEMPERATURE (IN K) TK AND PRESSURE (IN MPA) PMPA.  IN                C
C CONTRAST TO DENS, NO INITIAL GUESS IS REQUIRED (ONE IS GENERATED    C
C INTERNALLY).  THE VALUE OF DPD IS ALSO RETURNED.                     C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C DOUT    R    O   DENSITY AT TK AND PMPA, KG/M3                       C
C PMPA    R    I   PRESSURE, MPA                                       C
C TK      R    I   TEMPERATURE, K                                      C
C DPD     R    O   FIRST DERIVATIVE OF PMPA WITH RESPECT TO D AT DOUT  C
C IWORK   IA   -   INTEGER VECTOR FOR USE BY LOWER-LEVEL ROUTINES      C
C PROPR   RA   -   VECTOR OF REDUCED PROPERTIES USED BY LOWER-LEVEL    C
C                  ROUTINES                                            C
C IERR    I    O   DFIND0 RETURN STATUS FLAG.  NEGATIVES FROM DFIND1,  C
C                  POSITIVES FROM DFIND2.  MEANINGS:                   C
C                  0: CONVERGED                                        C
C                 -2: UNABLE TO BOUND ROOT (DFIND1)                    C
C                 -3: UNABLE TO CONVERGE BOUNDED ROOT (DFIND1)         C
C                  1: NO ROOT FOR REQUESTED PHASE.  RETURNS D WHERE    C
C                     DPD = 0 (DFIND2)                                 C
C                  2: UNABLE TO BOUND ROOT (DFIND2)                    C
C                  3: UNABLE TO CONVERGE BOUNDED ROOT (DFIND2)         C
C                  5: INPUT PRESSURE NOT IN VALID RANGE                C
C                  6: INPUT TEMPERATURE NOT IN VALID RANGE             C
C                  7: INPUT T,P TOO FAR IN SUBCOOLED SOLID REGION.     C
C                     CALCULATIONS ABORTED                             C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 07SEP95 - INITIAL CREATION BY AHH                                    C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 29SEP95 - AHH: CHANGE IWANT TO IWORK, REDOCUMENT ACCORDINGLY         C
C 19OCT95 - AHH: CHECK FOR VIOLATION OF ABSOLUTE T AND P LIMITS        C
C 15FEB96 - AHH: CHECK FOR TOO FAR INTO SUBCOOLED SOLID REGION         C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      INCLUDE 'wconst.cmn'
      INCLUDE 'wlimit.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWORK(NPROP)
C
C CHECK T AND P AGAINST LIMITS OF VALIDITY
C IF INVALID, RETURN A SMALL NUMBER
C
      IF ((PMPA .LT. PLOWER) .OR. (PMPA .GT. PUPPER)) THEN
        IERR = 5
        DOUT = 1.D-20
        DPD = 0.D0
        RETURN
      ENDIF
      IF ((TK .LT. TLOWER) .OR. (TK .GT. TUPPER)) THEN
        IERR = 6
        DOUT = 1.D-20
        DPD = 0.D0
        RETURN
      ENDIF
C
      IF (PMPA .LE. PTRIPW) THEN
        CALL TSUB(TSOLID, PMPA, IERS)
        IF ((IERS .NE. 1) .AND. (TK .LT. (TSOLID-35.D0))) THEN
          IERR = 7
          DOUT = 1.D-20
          DPD = 0.D0
          RETURN
        ENDIF
      ELSE
        CALL TMELT(TSOLID, PMPA, IFORM, IERS)
        IF ((IERS .NE. 2) .AND. (TK .LT. (TSOLID-35.D0))) THEN
          IERR = 7
          DOUT = 1.D-20
          DPD = 0.D0
          RETURN
        ENDIF
      ENDIF
C
      TR = TK / TCW
      PR = PMPA / PCW
      CALL DFIND0(DOUTR, PR, TR, DPDR, IWORK, PROPR, IERR)
      DOUT = DOUTR*RHOCW
      DPD = DPDR * PCW / RHOCW
      RETURN
      END
