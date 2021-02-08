      FUNCTION DLSAT(TR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** DLSAT ***                                    C
C THIS ROUTINE ESTIMATES THE REDUCED SAT. LIQUID DENSITY AT A GIVEN    C
C REDUCED TEMPERATURE T/TC USING THE EQUATION OF SAUL AND WAGNER (J.   C
C PHYS. CHEM. REF. DATA, 16, 893 (1987) [AS MODIFIED TO 1990 T-SCALE   C
C BY WAGNER AND PRUSS, J. PHYS. CHEM. REF. DATA, 22, 783 (1993)])      C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TR      R    I   REDUCED TEMPERATURE T/TC                            C
C DLSAT   R    O   REDUCED SATURATED LIQUID DENSITY RHO/RHOC           C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 17JUL95 - INITIAL CREATION BY AHH                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (E1=1.D0/3.D0, E2=2.D0/3.D0, E3=5.D0/3.D0,
     1           E4=16.D0/3.D0, E5=43.D0/3.D0, E6=110.D0/3.D0) 
      DIMENSION B(6)
      DATA B /1.99274064, 1.09965342, -0.510839303, -1.75493479,
     1        -45.5170352, -6.7469445D5/
      TAU = 1.D0 - TR
      DLSAT = 1.D0 + B(1)*TAU**E1 + B(2)*TAU**E2 + B(3)*TAU**E3 +
     1        B(4)*TAU**E4 + B(5)*TAU**E5 + B(6)*TAU**E6
      RETURN
      END
