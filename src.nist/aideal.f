      SUBROUTINE AIDEAL(LPHI, LPHID, LPHIT, LPHIDD, LPHITT, LPHIDT,
     1                  LPHDDD, LPHDDT, LPHDTT, PHI, PHID, PHIT,
     2                  PHIDD, PHITT, PHIDT, PHIDDD, PHIDDT, PHIDTT,
     3                  TAU, DEL)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** AIDEAL ***                                   C
C THIS ROUTINE COMPUTES THE REDUCED IDEAL-GAS HELMHOLTZ ENERGY AND ITS C
C (REDUCED) TEMPERATURE AND DENSITY DERIVATIVES FROM THE PRUSS AND     C
C WAGNER EOS FOR WATER AND STEAM.  ONLY THE QUANTITIES REQUESTED VIA   C
C THE LOGICAL ARGUMENTS ARE RETURNED.                                  C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C LXXXX   L    I   FOR WHETHER OR NOT TO COMPUTE QUANTITY XXXX         C
C PHI     R    O   REDUCED IDEAL-GAS HELMHOLTZ ENERGY                  C
C PHID    R    O   D(PHI)/D(DEL)                                       C
C PHIT    R    O   D(PHI)/D(TAU)                                       C
C PHIDD   R    O   D2(PHI)/D(DEL2)                                     C
C PHITT   R    O   D2(PHI)/D(TAU2)                                     C
C PHIDT   R    O   D2(PHI)/D(DEL)D(TAU)                                C
C PHIDDD  R    O   D3(PHI)/D(DEL3)                                     C
C PHIDDT  R    O   D3(PHI)/D(DEL2)D(TAU)                               C
C PHIDTT  R    O   D3(PHI)/D(DEL)D(TAU2)                               C
C TAU     R    I   DIMENSIONLESS INVERSE TEMPERATURE TC/T              C
C DEL     R    I   DIMENSIONLESS DENSITY RHO/RHOC                      C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 07JUL95 - INITIAL CREATION BY AHH                                    C
C 18AUG95 - MODIFIED VERSION AFTER DEBUGGING BY AHH AND JW             C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 16JAN97 - AHH: ADD THIRD DERIVATIVES                                 C
C 11FEB97 - AHH: MAKE EXP'S UNIFORMLY DOUBLE PRECISION                 C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      LOGICAL LPHI, LPHID, LPHIDD, LPHIT, LPHITT, LPHIDT
      LOGICAL LPHDDD, LPHDDT, LPHDTT
C (MUST INSERT COMMON CONTAINING WAGNER COEFFICIENTS)
      INCLUDE 'coefig.cmn'
C
C INITIALIZE OUTPUT QUANTITIES TO ZERO
C
      PHI = 0.D0
      PHID = 0.D0
      PHIDD = 0.D0
      PHIT = 0.D0
      PHITT = 0.D0
      PHIDT = 0.D0
      PHIDDD = 0.D0
      PHIDDT = 0.D0
      PHIDTT = 0.D0
C
C START WITH THE TERMS NOT IN THE SUM
C
      IF (LPHI) THEN
        PHI = LOG(DEL) + WA0(1) + WA0(2)*TAU + WA0(3)*LOG(TAU)
      ENDIF
      IF (LPHID) THEN
        PHID = 1.D0/DEL
      ENDIF
      IF (LPHIDD) THEN
        PHIDD = -1.D0/(DEL*DEL)
      ENDIF
      IF (LPHIT) THEN
        PHIT = WA0(2) + WA0(3)/TAU
      ENDIF
      IF (LPHITT) THEN
        PHITT = -WA0(3)/(TAU*TAU)
      ENDIF
      IF (LPHDDD) THEN
        PHIDDD = 2.D0 / (DEL**3)
      ENDIF
C
C NOTE NO PHIDT MODIFICATION BECAUSE THAT IS IDENTICALLY ZERO
C
C ADD IN TERMS FROM THE SUMMATION FOR PHI AND THE T DERIVATIVES
C AS NEEDED
C
      IF (LPHI .OR. LPHIT .OR. LPHITT) THEN
        DO 108 I = 4,8
          EGAM = DEXP(-WGAM0(I)*TAU)
          DIFF = 1.D0 - EGAM
          IF (LPHI) THEN
            PHI = PHI + WA0(I)*LOG(DIFF)
          ENDIF
          IF (LPHIT) THEN
            PHIT = PHIT + WA0(I)*WGAM0(I)*(1.D0/DIFF - 1.D0)
          ENDIF
          IF (LPHITT) THEN
            PHITT = PHITT - WA0(I)*WGAM0(I)*WGAM0(I)*EGAM/(DIFF*DIFF)
          ENDIF
  108   CONTINUE
      ENDIF
      RETURN
      END
