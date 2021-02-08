C
      SUBROUTINE TVAPC(PR, TS)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** TVAPC ***                                    C
C THIS ROUTINE ESTIMATES THE REDUCED TEMPERATURE T/TC AT A GIVEN       C
C REDUCED VAPOR PRESSURE P/PC NEAR THE CRITICAL POINT USING AN         C
C INTERPOLATION BETWEEN 647.095 K AND TC                               C
C IT ITERATIVELY CALLS PVAPC, WHICH CONTAINS THIS EQUATION             C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C PR      R    I   REDUCED SATURATION PRESSURE, PSAT/PC                C
C TS      R    O   REDUCED TEMPERATURE T/TC                            C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 10NOV99 - INITIAL CREATION BY AHH                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (TOL=1.D-10)
      PRLOG = LOG(PR)
C
C FOR INITIAL GUESS, USE GUESS FROM TVAP
C
      CALL TVAP(PR, TG)
  11  CONTINUE
      CALL PVAPC(TG, PSNEW, DPSDT)
      IF (DABS((PSNEW-PR)/PR) .LT. TOL) GO TO 777
      PERRLN = PRLOG - LOG(PSNEW)
      TGINV = 1./TG
      DLNPTI = -TG*TG*DPSDT/PSNEW
      TGINV2 = TGINV + PERRLN/DLNPTI
      TG = 1. / TGINV2
      IF (TG .GE. 1.0) TG = .999999
      GO TO 11      
  777 TS = TG
      RETURN
      END
