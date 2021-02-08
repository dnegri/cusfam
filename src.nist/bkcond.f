C
      SUBROUTINE BKCOND(TBAR, RBAR, COND0)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** BKCOND ***                                   C
C THIS ROUTINE COMPUTES THE BACKGROUND PART OF THE THERMAL CONDUCTIVITYC
C OF WATER AT A GIVEN TEMPERATURE AND DENSITY ACCORDING TO THE EQNS    C
C PRESENTED BY KESTIN ET AL., JPCRD, 13, 175 (1984).                   C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TBAR    R    I   TEMPERATURE MADE DIMENSIONLESS BY TSTAR             C
C RBAR    R    I   DENSITY MADE DIMENSIONLESS BY RSTAR                 C
C COND    R    O   BACKGROUND THERMAL CONDUCTIVITY (DIMENSIONLESS)     C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 25SEP95 - INITIAL ADAPTATION FROM OLD STEAM PROGRAM BY AHH           C
C 31JUL96 - AHH: CORRECT "A" ARRAY FOR PROPER USE OF REFERENCE CONST   C
C 11FEB97 - AHH: MAKE EXP'S UNIFORMLY DOUBLE PRECISION                 C
C 15MAY97 - AHH: MAKE COEFFICIENTS DOUBLE PRECISION                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION B(5,6)
      DATA B/1.3293046D0,1.7018363D0,5.2246158D0,8.7127675D0,
     1      -1.8525999D0,-0.40452437D0,
     1    -2.2156845D0,-10.124111D0,-9.5000611D0,.9340469D0,
     2     .24409490D0,1.6511057D0,4.9874687D0,4.3786606D0,0.D0,
     3       .018660751D0,-.76736002D0,-.27297694D0,-.91783782D0,0.D0,
     4     -.12961068D0,.37283344D0,-.43083393D0,0.D0,0.D0,
     5      .044809953D0,-.1120316D0,.13333849D0,0.D0,0.D0/
      DATA A0,A1,A2,A3/1.0D0,6.978267D0,2.599096D0,-0.998254D0/
C
      DR = RBAR - 1.D0
      TINV = 1.D0 / TBAR
      DT = TINV - 1.D0
C
      THCOND = A0+A1*TINV+A2*TINV**2+A3*TINV**3
      THCOND = DSQRT(TBAR)/THCOND
      SUM = 0.D0
      DO 10 I=1,5
      DO 10 J=1,6
   10 SUM = SUM+B(I,J)*DR**(J-1)*DT**(I-1)
      COND0 = THCOND*DEXP(RBAR*SUM)
      RETURN
      END
