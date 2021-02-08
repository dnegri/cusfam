C
      SUBROUTINE VISC(TBAR, RBAR, CHIBAR, VISC0, VISCBR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** VISC ***                                     C
C THIS ROUTINE COMPUTES THE VISCOSITY                                  C
C OF WATER AT A GIVEN TEMPERATURE AND DENSITY ACCORDING TO THE EQNS    C
C PRESENTED BY KESTIN ET AL., JPCRD, 13, 175 (1984).                   C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TBAR    R    I   TEMPERATURE MADE DIMENSIONLESS BY TSTAR             C
C RBAR    R    I   DENSITY MADE DIMENSIONLESS BY RSTAR                 C
C CHIBAR  R    I   DIMENSIONLESS ISOTHERMAL COMPRESSIBILITY            C
C VISC0   R    O   DIMENSIONLESS NORMAL PART OF VISCOSITY              C
C VISCBR  R    O   DIMENSIONLESS TOTAL VISCOSITY                       C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 25SEP95 - INITIAL ADAPTATION FROM OLD STEAM PROGRAM BY AHH           C
C 11FEB97 - AHH: MAKE EXP'S UNIFORMLY DOUBLE PRECISION                 C
C 15MAY97 - AHH: MAKE COEFFICIENTS DOUBLE PRECISION                    C
C 10SEP03 - AHH: REVISE BOUNDARIES ON CRITICAL TERM TO CONFORM TO NEW  C
C                IAPWS RELEASE                                         C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION A(4),B(6,7)
      DATA A/1.D0,.978197D0,.579829D0,-.202354D0/
      DATA B/.5132047D0,.3205656D0,2*0.D0,-.7782567D0,.1885447D0,
     1 .2151778D0,.7317883D0,1.241044D0,1.476783D0,2*0.D0,-.2818107D0,
     2 -1.070786D0,-1.263184D0,3*0.D0,.1778064D0,.4605040D0,.2340379D0,
     3 -.4924179D0,2*0.D0,-.0417661D0,2*0.D0, 
     4 .1600435D0,3*0.D0,-.01578386D0,7*0.D0,-.003629481D0,2*0.D0/
C
      TINV = 1.D0 / TBAR
      DT1 = TINV - 1.D0
      DR = RBAR - 1.D0
      SUM = 0.D0
      DO 10 K=1,4
 10   SUM = SUM+A(K)*TINV**(K-1) 
      VISCX = DSQRT(TBAR)/SUM
      SUM = 0.D0
      DO 20 I=1,6
      DT1I = DT1**(I-1)
      DO 20 J=1,7
 20   SUM = SUM+B(I,J)*DT1I*DR**(J-1)
      VISC0 = VISCX*DEXP(RBAR*SUM)
C
      IF (CHIBAR.LT.21.93D0) THEN
        VISCBR = VISC0
      ELSE
        V2=.922D0*CHIBAR**(.0263D0)
        VISCBR = VISC0*V2
      ENDIF
      RETURN
      END
