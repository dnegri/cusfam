C
      FUNCTION DVSAT(TR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** DVSAT ***                                    C
C THIS ROUTINE ESTIMATES THE REDUCED SAT. VAPOR DENSITY AT A GIVEN     C
C REDUCED TEMPERATURE T/TC USING THE EQUATION OF SAUL AND WAGNER (J.   C
C PHYS. CHEM. REF. DATA, 16, 893 (1987) [AS MODIFIED TO 1990 T-SCALE   C
C BY WAGNER AND PRUSS, J. PHYS. CHEM. REF. DATA, 22, 783 (1993)])      C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TR      R    I   REDUCED TEMPERATURE T/TC                            C
C DVSAT   R    O   REDUCED SATURATED VAPOR DENSITY RHO/RHOC            C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 17JUL95 - INITIAL CREATION BY AHH                                    C
C 11FEB97 - AHH: MAKE EXP'S UNIFORMLY DOUBLE PRECISION                 C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (E1=1.D0/3.D0, E2=2.D0/3.D0, E3=4.D0/3.D0,
     1           E4=3.D0, E5=37.D0/6.D0, E6=71.D0/6.D0) 
      DIMENSION C(6)
      DATA C /-2.03150240D0, -2.68302940D0, -5.38626492D0, 
     1        -17.2991605D0, -44.7586581D0, -63.9201063D0/
      TAU = 1.D0 - TR
      DVSAT =  C(1)*TAU**E1 + C(2)*TAU**E2 + C(3)*TAU**E3 +
     1        C(4)*TAU**E4 + C(5)*TAU**E5 + C(6)*TAU**E6
      DVSAT = DEXP(DVSAT)
      RETURN
      END
