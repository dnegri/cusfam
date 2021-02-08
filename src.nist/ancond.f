C
      SUBROUTINE ANCOND(TBAR, RBAR, CHIBAR, DPDTBR, VBAR0, COND1)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** ANCOND ***                                   C
C THIS ROUTINE COMPUTES THE ANOMALOUS PART OF THE THERMAL CONDUCTIVITY C
C OF WATER AT A GIVEN TEMPERATURE AND DENSITY ACCORDING TO THE EQNS    C
C PRESENTED BY KESTIN ET AL., JPCRD, 13, 175 (1984).                   C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TBAR    R    I   TEMPERATURE MADE DIMENSIONLESS BY TSTAR             C
C RBAR    R    I   DENSITY MADE DIMENSIONLESS BY RSTAR                 C
C CHIBAR  R    I   DIMENSIONLESS ISOTHERMAL COMPRESSIBILITY            C
C DPDTBR  R    I   DIMENSIONLESS DP/DT                                 C
C VBAR0   R    I   DIMENSIONLESS NORMAL PART OF VISCOSITY              C
C COND1   R    O   ANOMALOUS THERMAL CONDUCTIVITY (DIMENSIONLESS)      C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 25SEP95 - INITIAL ADAPTATION FROM OLD STEAM PROGRAM BY AHH           C
C 31JUL96 - AHH: STREAMLINE CODE                                       C
C 25SEP96 - AHH: PREVENT UNDERFLOW                                     C
C 11FEB97 - AHH: MAKE EXP'S UNIFORMLY DOUBLE PRECISION                 C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DATA A,C,OMEGA/18.66D0,1.3848D-03,0.4678D0/
C
      DT = TBAR - 1.D0
      DR = RBAR - 1.D0
      IF (DPDTBR .LT. 0.D0) DPDTBR = 0.D0
      CHIR = (DABS(CHIBAR))**OMEGA
      EX = -A*DT**2 - DR**4
      IF (EX .LT. -600.D0) THEN
        COND1 = 0.D0
      ELSE
        FAC = (TBAR/RBAR*DPDTBR)**2 * CHIR * DSQRT(RBAR)
        COND1 = C*FAC*DEXP(EX)/VBAR0
      ENDIF
      RETURN
      END
