C
      SUBROUTINE PSAT(TK, PMPA, RHOL, RHOV, IWORK, PROPR, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PSAT ***                                     C
C THIS ROUTINE COMPUTES THE COEXISTENCE PRESSURE (IN MPA) AT A         C
C GIVEN TEMPERATURE (IN K).  IT ALSO RETURNS THE COEXISTING            C
C DENSITIES.  IT CALLS PCOEX, WHICH WORKS IN REDUCED UNITS.            C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    I   TEMPERATURE, K                                      C
C PMPA    R    O   COEXISTENCE PRESSURE, MPA                           C
C RHOL    R    O   SATURATED LIQUID DENSITY, KG/M3                     C
C RHOV    R    O   SATURATED VAPOR DENSITY, KG/M3                      C
C IWORK   IA   -   INTEGER VECTOR FOR USE BY LOWER-LEVEL ROUTINES      C
C PROPR   RA   -   VECTOR OF REDUCED PROPERTIES USED BY LOWER-LEVEL    C
C                  ROUTINES                                            C
C IERR    I    O   ERROR FLAG (FROM PCOEX)                             C
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
C 07SEP95 - INITIAL CREATION BY AHH                                    C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 29SEP95 - AHH: CHANGE IWANT TO IWORK, REDOCUMENT ACCORDINGLY         C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 13JAN97 - AHH: DOCUMENT IERR=4                                       C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      INCLUDE 'wconst.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWORK(NPROP)
      TR = TK/TCW
      CALL PCOEX(TR, PR, RHOLR, RHOVR, IWORK, PROPR, IERR)
      PMPA = PR*PCW
      RHOV = RHOVR*RHOCW
      RHOL = RHOLR*RHOCW
      RETURN
      END
