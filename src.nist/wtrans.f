C
      SUBROUTINE WTRANS(MODE, TK, RHO, CHI, DPDT, THCOND, VISCOS)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** WTRANS ***                                   C
C THIS ROUTINE COMPUTES THE THERMAL CONDUCTIVITY AND VISCOSITY OF      C
C WATER AT A GIVEN TEMPERATURE AND DENSITY ACCORDING TO THE EQUATIONS  C
C PRESENTED BY KESTIN ET AL., JPCRD, 13, 175 (1984).                   C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C MODE    I    I   FOR WHETHER OR NOT TO COMPUTE VARIOUS PROPERTIES,   C
C                  1=THCOND ONLY, 2=VISC ONLY, 3=BOTH                  C
C TK      R    I   TEMPERATURE IN KELVINS                              C
C RHO     R    I   DENSITY IN KG/M3                                    C
C CHI     R    I   ISOTHERMAL COMPRESSIBILITY IN MPA**(-1)             C
C DPDT    R    I   DPDT IN MPA/K                                       C
C THCOND  R    O   THERMAL CONDUCTIVITY IN W/m.K                       C
C VISCOS  R    O   VISCOSITY IN PA.S                                   C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 25SEP95 - INITIAL CREATION BY AHH                                    C
C 12AUG96 - CORRECT CALCULATION OF CHIBAR                              C
C 18SEP96 - AHH: CONVERT TREF TO 1990 TEMPERATURE SCALE (WAS 647.27)   C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C REFERENCE CONSTANTS FROM KESTIN ET AL.
C
      DATA TREF, RHOREF, PREF, CNDREF, VREF
     >     /647.226D0, 317.763D0, 22.115D0, .4945D0, 55.071D-6/
C
C COMPUTE "REDUCED" TEMPERATURE AND DENSITY AND COMPR.
C
      TBAR  = TK / TREF
      RBAR = RHO / RHOREF
      CHIBAR = CHI * PREF * RBAR**2
C
C COMPUTE VISCOSITY ALWAYS, SINCE NEEDED FOR THCOND ALSO
C
      CALL VISC (TBAR, RBAR, CHIBAR, VISC0, VISCBR)
      VISCOS = VISCBR*VREF
C
C COMPUTE THERMAL CONDUCTIVITY IF REQUESTED
C
      IF (MOD(MODE,2) .EQ. 1) THEN
C
C COMPUTE REDUCED DPDT, THEN CALL FOR BACKGROUND PART
C
        DPDTBR = DPDT*TREF/PREF
        CALL BKCOND (TBAR, RBAR, COND0)
C
C CALL FOR ANOMALOUS PART OF CONDUCTIVITY UNLESS RHO IS HIGH
C
        IF (RHO/RHOREF .LT. 3.D0) THEN
           CALL ANCOND (TBAR, RBAR, CHIBAR, DPDTBR, VISC0, COND1)
           CONDBR = COND0 + COND1
        ELSE
           CONDBR = COND0
        ENDIF
        THCOND = CONDBR*CNDREF
      ENDIF
      RETURN
      END
