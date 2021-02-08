C
      SUBROUTINE PVAP(TR, PS, DPSDT)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PVAP ***                                     C
C THIS ROUTINE ESTIMATES THE REDUCED VAPOR PRESSURE P/PC AT A GIVEN    C
C REDUCED TEMPERATURE T/TC USING THE EQUATION OF SAUL AND WAGNER (J.   C
C PHYS. CHEM. REF. DATA, 16, 893 (1987) [AS MODIFIED TO 1990 T-SCALE   C
C BY WAGNER AND PRUSS, J. PHYS. CHEM. REF. DATA, 22, 783 (1993)])      C
C THE FIRST DERIVATIVE WITH RESPECT TO TR IS ALSO RETURNED.            C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TR      R    I   REDUCED TEMPERATURE T/TC                            C
C PS      R    O   REDUCED SATURATION PRESSURE, PSAT/PC                C
C DPSDT   R    O   FIRST DERIVATIVE OF PS WITH RESPECT TO TR           C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 17JUL95 - INITIAL CREATION BY AHH                                    C
C 11FEB97 - AHH: MAKE EXP'S UNIFORMLY DOUBLE PRECISION                 C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION A(6)
      DATA A /-7.85951783D0, 1.84408259D0, -11.7866497D0, 
     1        22.6807411D0, -15.9618719D0, 1.80122502D0/
      TAU = 1.D0 - TR
      SUM = TAU * (A(1) + A(2)*DSQRT(TAU) + A(3)*TAU*TAU + 
     1      A(4)*TAU**2.5D0 + A(5)*TAU**3 + A(6)*TAU**6.5D0)
      PSLN = SUM / TR
      PS = DEXP(PSLN)
      DSDTAU = A(1) + 1.5D0*A(2)*DSQRT(TAU) + 3.D0*A(3)*TAU*TAU +
     1         3.5D0*A(4)*TAU**2.5D0 + 4.D0*A(5)*TAU**3 + 
     2         7.5D0*A(6)*TAU**6.5D0
      DLNPDT = -(PSLN + DSDTAU) / TR
      DPSDT = PS*DLNPDT
      RETURN
      END
