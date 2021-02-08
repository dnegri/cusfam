C
      SUBROUTINE PVAPC(TR, PR, DPDT)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PVAPC ***                                    C
C THIS ROUTINE ESTIMATES THE REDUCED SATURATION PRESSURE AT A GIVEN    C
C REDUCED TEMPERATURE T/TC USING AN INTERPOLATION (LN P VS. 1/T)       C
C BETWEEN THE KNOWN VALUES AT 647.095 K AND TC                         C
C SHOULD ONLY BE CALLED FOR TR .GT. 0.999998                           C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TR      R    I   REDUCED TEMPERATURE T/TC                            C
C PR      R    O   REDUCED SATURATION PRESSURE PSAT/PC                 C
C DPDT    R    O   D(PR)/D(TR) (SLOPE OF REDUCED VAPOR PRESSURE CURVE) C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 10NOV99 - INITIAL CREATION BY AHH                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (A1 = 7.83928D0)
      THING = (TR - 1.D0) / TR
      PR = DEXP(A1*THING)
      DPDT = A1*PR/(TR*TR)
      RETURN
      END
