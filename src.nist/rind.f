C
      SUBROUTINE RIND(TK, RHO, ALAM, RINDX, IERLAM, N)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** RIND  ***                                    C
C THIS SUBROUTINE COMPUTES THE REFRACTIVE INDEX OF WATER AT A          C
C GIVEN TEMPERATURE, DENSITY, AND ARRAY OF WAVELENGTHS                 C
C ACCORDING TO THE REFIT EQUATION OF SCHIEBENER ET AL                  C
C IT TAKES AN ARRAY OF WAVELENGTHS AND CALLS RIND2 FOR EACH OF THEM    C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    I   TEMPERATURE IN KELVINS                              C
C RHO     R    I   DENSITY IN KG/M3                                    C
C ALAM    RA   I   WAVELENGTHS (MICROMETERS) AT WHICH TO COMPUTE RINDX C
C RINDX   RA   O   REFRACTIVE INDEX AT ALAM VALUES                     C
C IERLAM  IA   O   ARRAY OF RETURN FLAGS FOR WAVELENGTHS               C
C                  0     = OK                                          C
C                  1(-1) = LARGER(SMALLER) THAN RECOMMENDED RANGE,     C
C                          VALUE IS AN EXTRAPOLATION                   C
C                  2(-2) = TOO LARGE(SMALL), BEYOND REASONABLE         C
C                          EXTRAPOLATION RANGE. WILL STILL RETURN      C
C                          COMPUTED NUMBER IF WITHIN RANGE WHERE CALC. C
C                          DOESN'T BLOW UP, OTHERWISE RETURNS ZERO.    C
C N       I    I   LENGTH OF ALAM, RINDX, AND IERLAM ARRAYS            C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 19AUG96 - INITIAL IMPLEMENTATION BY AHH                              C
C 25MAR97 - AHH: CHANGES FOR CALLING UNDER PROPS0                      C
C 27MAY97 - AHH: CHANGE ERROR/WARNING HANDLING                         C
C 22SEP97 - AHH: ADJUST UPPER EXTRAPOLATION LIMIT TO 1.9 IN ACCORDANCE C
C                WITH IAPWS RELEASE                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (TREF=273.15D0, RREF = 1.D3, ALREF=0.589D0)
      PARAMETER (ABIG=1.1D0, ASMALL=0.2D0, BIG2=1.9D0, SMALL2=0.18D0)
      DIMENSION ALAM(N), RINDX(N), IERLAM(N)
C
      TBAR = TK / TREF
      RBAR = RHO / RREF
C
C - LOOP THROUGH EACH POINT, CALLING RIND2 FOR CALCULATION
C
      DO 100 I=1,N
        ALAMBR = ALAM(I) / ALREF
        CALL RIND2(ALAMBR, RBAR, TBAR, RINDX(I))
C
C - CHECK TO SEE IF OUTSIDE RECOMMENDED BOUNDS
C
        IF (ALAM(I) .LT. SMALL2) THEN
          IERLAM(I) = -2
        ELSE IF (ALAM(I) .GT. BIG2) THEN
          IERLAM(I) = 2
        ELSE IF (ALAM(I) .LT. ASMALL) THEN
          IERLAM(I) = -1
        ELSE IF (ALAM(I) .GT. ABIG) THEN
          IERLAM(I) = 1
        ELSE
          IERLAM(I) = 0
        ENDIF
  100 CONTINUE
C
      RETURN
      END
