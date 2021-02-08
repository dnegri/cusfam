C
      SUBROUTINE TVAP(PR, TS)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** TVAP ***                                     C
C THIS ROUTINE ESTIMATES THE REDUCED TEMPERATURE T/TC AT A GIVEN       C
C REDUCED VAPOR PRESSURE P/PC USING THE EQUATION OF SAUL AND WAGNER (J.C
C PHYS. CHEM. REF. DATA, 16, 893 (1987) [AS MODIFIED TO 1990 T-SCALE   C
C BY WAGNER AND PRUSS, J. PHYS. CHEM. REF. DATA, 22, 783 (1993)])      C
C IT ITERATIVELY CALLS PVAP, WHICH CONTAINS THIS EQUATION              C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C PR      R    I   REDUCED SATURATION PRESSURE, PSAT/PC                C
C TS      R    O   REDUCED TEMPERATURE T/TC                            C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 17JUL95 - INITIAL CREATION BY AHH                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (TOL=1.D-7)
      PRLOG = LOG(PR)
C
C FOR INITIAL GUESS, USE GUESS FROM HGK ROUTINE TSAT
C
      PL=7.1226152+2.302585*DLOG(PR)
      TG=.57602+PL*(.042887+PL*(.00368+PL*(3.837E-4+PL*3.E-5)))
      IF(TG.LT..422) TG=.422
      IF(TG.GT..9996) TG=.9996
  11  CONTINUE
      CALL PVAP(TG, PSNEW, DPSDT)
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
