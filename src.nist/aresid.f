C
      SUBROUTINE ARESID(LPHI, LPHID, LPHIT, LPHIDD, LPHITT, LPHIDT,
     1                  LPHDDD, LPHDDT, LPHDTT, PHI, PHID, PHIT,
     2                  PHIDD, PHITT, PHIDT, PHIDDD, PHIDDT, PHIDTT,
     3                  TAU, DEL)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** ARESID ***                                   C
C THIS ROUTINE COMPUTES THE REDUCED RESIDUAL HELMHOLTZ ENERGY AND ITS  C
C (REDUCED) TEMPERATURE AND DENSITY DERIVATIVES FROM THE PRUSS AND     C
C WAGNER EOS FOR WATER AND STEAM.  ONLY THE QUANTITIES REQUESTED VIA   C
C THE LOGICAL ARGUMENTS ARE RETURNED.                                  C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C LXXXX   L    I   FOR WHETHER OR NOT TO COMPUTE QUANTITY XXXX         C
C PHI     R    O   REDUCED RESIDUAL HELMHOLTZ ENERGY                   C
C PHID    R    O   D(PHI)/D(DEL)                                       C
C PHIT    R    O   D(PHI)/D(TAU)                                       C
C PHIDD   R    O   D2(PHI)/D(DEL2)                                     C
C PHITT   R    O   D2(PHI)/D(TAU2)                                     C
C PHIDT   R    O   D2(PHI)/D(DEL)D(TAU)                                C
C PHIDDD  R    O   D3(PHI)/D(DEL3)                                     C
C PHIDDT  R    O   D3(PHI)/D(DEL2)D(TAU)                               C
C PHIDTT  R    O   D3(PHI)/D(DEL)D(TAU2)                               C
C TAU     R    I   DIMENSIONLESS RECIPROCAL TEMPERATURE TC/T           C
C DEL     R    I   DIMENSIONLESS DENSITY RHO/RHOC                      C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 06JUL95 - INITIAL CREATION BY AHH                                    C
C 18AUG95 - MODIFIED VERSION AFTER DEBUGGING BY AHH AND JW             C
C 19OCT95 - AHH: SPEED ENHANCEMENTS                                    C
C 21JUN96 - AHH: CORRECT DOCUMENTATION OF TAU                          C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 25SEP96 - AHH: AVOID UNDERFLOWS                                      C
C 16JAN97 - AHH: ADD THIRD DERIVATIVES                                 C
C 11FEB97 - AHH: MAKE EXP'S UNIFORMLY DOUBLE PRECISION                 C
C 21MAR97 - AHH: CORRECT ERROR IN PHIDDD                               C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      LOGICAL LPHI, LPHID, LPHIDD, LPHIT, LPHITT, LPHIDT
      LOGICAL LPHDDD, LPHDDT, LPHDTT
C
C (MUST INSERT COMMON CONTAINING WAGNER COEFFICIENTS)
      INCLUDE 'coef.cmn'
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
      DINV = 1.D0 / DEL
      DINV2 = DINV*DINV
      DINV3 = DINV2*DINV
      TINV = 1.D0 / TAU
      TINV2 = TINV*TINV
C
C COMPUTE THE FIRST SEVEN TERMS IN THE SUM FOR ANY OF THE RESIDUAL
C PROPERTY DERIVATIVES THAT ARE NEEDED
C
      DO 107 I = 1,7
        DELPOW = DEL**IWD(I)
        TAUPOW = TAU**WT(I)
        WNDT = WN(I)*DELPOW*TAUPOW          
        IF (LPHI) THEN
          PHI = PHI + WNDT
        ENDIF
        IF (LPHID) THEN
          PHID = PHID + WNDT*IWD(I)*DINV
        ENDIF
        IF (LPHIDD) THEN
          PHIDD = PHIDD + WNDT*IWD(I)*(IWD(I)-1)*DINV2
        ENDIF
        IF (LPHIT) THEN
          PHIT = PHIT + WNDT*WT(I)*TINV
        ENDIF
        IF (LPHITT) THEN
          PHITT = PHITT + WNDT*WT(I)*(WT(I)-1.D0)*TINV2
        ENDIF
        IF (LPHIDT) THEN
          PHIDT = PHIDT + WNDT*IWD(I)*DINV*WT(I)*TINV
        ENDIF
        IF (LPHDDD) THEN
          PHIDDD = PHIDDD + WNDT*IWD(I)*(IWD(I)-1)*(IWD(I)-2)*DINV3
        ENDIF
        IF (LPHDDT) THEN
          PHIDDT = PHIDDT + WNDT*IWD(I)*(IWD(I)-1)*WT(I)*TINV*DINV2
        ENDIF
        IF (LPHDTT) THEN
          PHIDTT = PHIDTT + WNDT*IWD(I)*WT(I)*(WT(I)-1.D0)*TINV2*DINV
        ENDIF
C
  107 CONTINUE
C
C NOW DO THE SAME THING FOR TERMS 8 THROUGH 51
C
      DO 151 I = 8,51
        DELPOW = DEL**IWD(I)
        TAUPOW = TAU**IWT(I)
        DELC = DEL**IWC(I)          
        IF (DELC .GT. 700.D0) THEN
          EPOW = 0.D0
        ELSE
          EPOW = DEXP(-DELC)
        ENDIF
        WNDTE = WN(I)*DELPOW*TAUPOW*EPOW
        IF (LPHI) THEN
          PHI = PHI + WNDTE
        ENDIF
        IF (LPHID) THEN
          PHID = PHID + WNDTE*DINV*(IWD(I)-IWC(I)*DELC)
        ENDIF
        IF (LPHIDD) THEN
          PROD1 = IWC(I)*DELC
          PHIDD = PHIDD + WNDTE*DINV2*
     1                    ((IWD(I)-PROD1)*(IWD(I)-1.D0-PROD1)
     2                     -IWC(I)*PROD1)
        ENDIF
        IF (LPHIT) THEN
          PHIT = PHIT + WNDTE*IWT(I)*TINV
        ENDIF
        IF (LPHITT) THEN
          PHITT = PHITT + WNDTE*IWT(I)*(IWT(I)-1)*TINV2
        ENDIF
        IF (LPHIDT) THEN
          PHIDT = PHIDT + WNDTE*DINV*IWT(I)*TINV*
     1                    (IWD(I)-IWC(I)*DELC)
        ENDIF
        IF (LPHDDD) THEN
          IC2 = IWC(I)*IWC(I)
          IC3 = IC2*IWC(I)
          IS1 = -2*IWC(I) + 3*IC2 - IC3 + 6*IWC(I)*IWD(I)
     1         - 3*IWD(I)*(IC2+IWC(I)*IWD(I))
          IS2 = 3 * (IC3 - IC2 + IC2*IWD(I))
          PHIDDD = PHIDDD + WNDTE*DINV3*
     1             (IWD(I)*(2-3*IWD(I)+IWD(I)*IWD(I)) + IS1*DELC
     2              + IS2*DELC*DELC - IC3*DELC**3)
        ENDIF
        IF (LPHDDT) THEN
          IC2 = IWC(I)*IWC(I)
          IS1 = IWC(I) - IC2 - 2*IWC(I)*IWD(I)
          PHIDDT = PHIDDT + WNDTE*IWT(I)*TINV*DINV2*
     1             (IWD(I)*IWD(I) - IWD(I) + IS1*DELC + IC2*DELC*DELC)
        ENDIF
        IF (LPHDTT) THEN
          PHIDTT = PHIDTT + WNDTE*IWT(I)*(IWT(I)-1)*TINV2*DINV*
     1                      (IWD(I) - IWC(I)*DELC)
        ENDIF
C
  151 CONTINUE
C
C NOW DO THE SAME THING FOR TERMS 52 THROUGH 54
C
      DO 154 I = 52,54
        DELPOW = DEL**IWD(I)
        TAUPOW = TAU**IWT(I)
        DEPS = DEL - IWEPS(I)
        TGAM = TAU - WGAM(I)
        ALFEXP = IWALF(I)*DEPS*DEPS
        BETEXP = IWBET(I)*TGAM*TGAM
        ALFBET = ALFEXP + BETEXP
        IF (ALFBET .GT. 700.D0) THEN
          EPOW = 0.D0
        ELSE
          EPOW = DEXP(-ALFBET)
        ENDIF
        WNDTE = WN(I)*DELPOW*TAUPOW*EPOW
        IF (LPHI) THEN
          PHI = PHI + WNDTE
        ENDIF
        IF (LPHID) THEN
          PHID = PHID + WNDTE*(IWD(I)*DINV - 2.D0*IWALF(I)*DEPS)
        ENDIF
        IF (LPHIDD) THEN
          PHIDD = PHIDD + WNDTE * (-2.D0*IWALF(I) +
     2                     4.D0*IWALF(I)**2*DEPS*DEPS -
     3                     4.D0*IWD(I)*IWALF(I)*DINV*DEPS +
     4                     IWD(I)*(IWD(I)-1)*DINV2)
        ENDIF
        IF (LPHIT) THEN
          PHIT = PHIT + WNDTE * (IWT(I)*TINV - 2.D0*IWBET(I)*TGAM)
        ENDIF
        IF (LPHITT) THEN
          PHITT = PHITT + WNDTE *
     1                    ((IWT(I)*TINV - 2.D0*IWBET(I)*TGAM)**2
     2                     - IWT(I)*TINV2 - 2.D0*IWBET(I))
        ENDIF
        IF (LPHIDT) THEN
          PHIDT = PHIDT + WNDTE *
     1                    (IWD(I)*DINV - 2.D0*IWALF(I)*DEPS) *
     2                    (IWT(I)*TINV - 2.D0*IWBET(I)*TGAM)
        ENDIF
        IF (LPHDDD) THEN
          IA2E = IWALF(I)*IWALF(I)*IWEPS(I)
          IAE2 = IWALF(I)*IWEPS(I)*IWEPS(I)
          IS1 = 6*IWALF(I)*IWD(I)*IWEPS(I) * (IWD(I)-1)
          IS2 = 6*IWALF(I)*IWD(I) * (2*IAE2-IWD(I))
          IS3 = 4*IA2E * (2*IAE2-6*IWD(I)-3)
          IS4 = 12*IWALF(I)*IWALF(I) * (1+IWD(I)-2*IAE2)
          PHIDDD = PHIDDD + WNDTE * DINV3 *
     1             (IWD(I)*(2-3*IWD(I)+IWD(I)*IWD(I)) + IS1*DEL
     2              + IS2*DEL**2 + IS3*DEL**3 + IS4*DEL**4
     3              + 24*IWALF(I)*IA2E*DEL**5 - 8*IWALF(I)**3*DEL**6)
        ENDIF
        IF (LPHDDT) THEN
          IS2 = 2*IWALF(I) * (2*IWALF(I)*IWEPS(I)**2 - 2*IWD(I) - 1)
          PHIDDT = PHIDDT + WNDTE * DINV2 * TINV *
     1             (IWD(I)*(IWD(I)-1) + 4*IWALF(I)*IWD(I)*IWEPS(I)*DEL
     2              + IS2*DEL**2 - 8*IWALF(I)**2*IWEPS(I)*DEL**3
     3              + 4*IWALF(I)*IWALF(I)*DEL**4) *
     4             (IWT(I) + 2*IWBET(I)*TAU*(WGAM(I)-TAU))
        ENDIF
        IF (LPHDTT) THEN
          S2 = 2*IWBET(I) * (2*IWBET(I)*WGAM(I)**2 - 2*IWT(I) - 1)
          PHIDTT = PHIDTT + WNDTE * DINV * TINV2 *
     1             (IWT(I)*(IWT(I)-1) + 4*IWBET(I)*WGAM(I)*IWT(I)*TAU
     2              + S2*TAU**2 - 8*IWBET(I)**2*WGAM(I)*TAU**3
     3              + 4*IWBET(I)*IWBET(I)*TAU**4) *
     4             (IWD(I) + 2*IWALF(I)*DEL*(IWEPS(I)-DEL))
        ENDIF
C
  154 CONTINUE
C
C NOW FOR THE I=55 AND 56 TERMS
C
      Q = (1.D0-DEL)**2
      DM1 = DEL - 1.D0
      TM1 = TAU - 1.D0
      DO 156 I=55,56
        PEXP = -IWCC(I)*Q - IWDD(I)*TM1**2
        IF (PEXP .LT. -700.D0) THEN
          PSI = 0.D0
        ELSE
          PSI = DEXP(PEXP)
        ENDIF
        THETA = 1.D0 - TAU + WAA(I)*Q**(0.5D0/WBET(I))
        DELTA = THETA*THETA + WBB(I)*Q**WA(I)
        BM1 = WB(I) - 1.D0
        BM2 = BM1 - 1.D0
        BM3 = BM2 - 1.D0
        AM1 = WA(I) - 1.D0
        AM2 = AM1 - 1.D0
        AM3 = AM2 - 1.D0
        DB = DELTA**WB(I)
        IF (DELTA.EQ.0.D0) THEN
          DBM1 = 0.D0
          DBM2 = 0.D0
          DBM3 = 0.D0
        ELSE
          DBM1 = DELTA**BM1
          DBM2 = DELTA**BM2
          DBM3 = DELTA**BM3
        ENDIF
        IF (LPHI) THEN
          PHI = PHI + WN(I)*DB*DEL*PSI
        ENDIF
C
C CERTAIN CALCULATIONS ARE NEEDED WHEN THERE IS AT LEAST ONE 
C DIFFERENTIATION WITH RESPECT TO TAU
C
        IF (LPHIT .OR. LPHITT .OR. LPHIDT .OR. LPHDDT .OR. LPHDTT) THEN
          DDDT = -2.D0*THETA
          DDBDT = WB(I)*DBM1*DDDT
          DPDT = -2.D0*IWDD(I)*TM1*PSI
        ENDIF
C
C CERTAIN CALCULATIONS ARE NEEDED WHEN THERE IS AT LEAST ONE 
C DIFFERENTIATION WITH RESPECT TO DEL
C
        IF (LPHID .OR. LPHIDD .OR. LPHIDT .OR. LPHDDD .OR. LPHDDT .OR.
     1      LPHDTT) THEN
          DQDD = 2.D0*DM1
          OVERB = 0.5D0/WBET(I)
          OBM1 = OVERB - 1.D0
          DTHDD = WAA(I)*OVERB * Q**OBM1 * DQDD
          DDDD = 2.D0*THETA*DTHDD + WBB(I)*WA(I)*Q**AM1*DQDD
          DDBDD = WB(I)*DBM1*DDDD
          DPDD = -2.D0*IWCC(I)*DM1*PSI
        ENDIF
C
C CERTAIN CALCULATIONS ARE NEEDED WHEN THERE IS AT LEAST ONE 
C DIFFERENTIATION WITH RESPECT TO DEL AND ONE W.R.T. TAU
C
        IF (LPHIDT .OR. LPHDDT .OR. LPHDTT) THEN
          D2DBTD = -2.D0*WB(I)*(THETA*BM1*DBM2*DDDD + DTHDD*DBM1)
          D2PDTD = 4.D0*IWCC(I)*IWDD(I)*DM1*TM1*PSI
        ENDIF
C
C CERTAIN CALCULATIONS ARE NEEDED WHEN THERE ARE AT LEAST TWO 
C DIFFERENTIATIONS WITH RESPECT TO DEL
C
        IF (LPHIDD .OR. LPHDDD .OR. LPHDDT) THEN
          OBM2 = OBM1 - 1.D0
          IF (Q.EQ.0.D0) THEN
            D2THDD = 0.D0
            D2DDD = 2.D0*(THETA*D2THDD + DTHDD*DTHDD)
          ELSE
            D2THDD = WAA(I)*OVERB*(OBM1*Q**OBM2*DQDD*DQDD +
     1             2.D0*Q**OBM1)
            D2DDD = 2.D0*(THETA*D2THDD + DTHDD*DTHDD) + WBB(I)*WA(I)*
     1            (AM1*Q**AM2*DQDD*DQDD + 2.D0*Q**AM1)
          ENDIF
          D2DBDD = WB(I)*(BM1*DBM2*DDDD*DDDD + DBM1*D2DDD)
          D2PDD = 2.D0*IWCC(I)*PSI*(2.D0*IWCC(I)*Q - 1.D0)
        ENDIF
C
C CERTAIN CALCULATIONS ARE NEEDED WHEN THERE ARE AT LEAST TWO 
C DIFFERENTIATIONS WITH RESPECT TO TAU
C
        IF (LPHITT .OR. LPHDTT) THEN
          D2DBDT = 2.D0*WB(I)*(2.D0*THETA*THETA*BM1*DBM2 + DBM1)
          D2PDT = 2.D0*IWDD(I)*PSI*(2.D0*IWDD(I)*TM1*TM1 - 1.D0)
        ENDIF
C
        IF (LPHID) THEN
          PHID = PHID + WN(I)*(DB*(PSI+DEL*DPDD)+DDBDD*DEL*PSI)
        ENDIF
        IF (LPHIDD) THEN
          PHIDD = PHIDD + WN(I)*(DB*(2.D0*DPDD+DEL*D2PDD) +
     1            2.D0*DDBDD*(PSI+DEL*DPDD) + D2DBDD*DEL*PSI)
        ENDIF
        IF (LPHIT) THEN
          PHIT = PHIT + WN(I)*DEL*(DDBDT*PSI+DB*DPDT)
        ENDIF
        IF (LPHITT) THEN
          PHITT = PHITT + WN(I)*DEL*(D2DBDT*PSI + 2.D0*DDBDT*DPDT +
     1            DB*D2PDT)
        ENDIF
        IF (LPHIDT) THEN
          PHIDT = PHIDT + WN(I)*(DB*(DPDT+DEL*D2PDTD) + DEL*DDBDD*DPDT
     1            + DDBDT*(PSI+DEL*DPDD) + D2DBTD*DEL*PSI)
        ENDIF
        IF (LPHDDD) THEN
          IF (Q.EQ.0.D0) THEN
            D3THDD = 0.D0
            D3DDD = 0.D0
          ELSE
            D3THDD = WAA(I)*OVERB*OBM1 * (OBM2*Q**(OVERB-3.D0)*DQDD**3
     1                                    + 6.D0*Q**OBM2*DQDD)
            D3DDD = 6.D0*DTHDD*D2THDD + 2.D0*THETA*D3THDD +
     1              WBB(I)*WA(I)*AM1 * (AM2*Q**AM3*DQDD**3 +
     2                                  6.D0*Q**AM2*DQDD)
          ENDIF
          D3DBDD = WB(I) * (BM1*(BM2*DBM3*DDDD**3+3.D0*DBM2*DDDD*D2DDD)
     1                      + DBM1*D3DDD)
          D3PDD = 4*IWCC(I)**2*DM1*PSI * (3-2*IWCC(I)*DM1**2)
          PHIDDD = PHIDDD + WN(I) * (DB*(3.D0*D2PDD+DEL*D3PDD)
     1                             + 3.D0*DDBDD*(2.D0*DPDD+DEL*D2PDD)
     2                             + 3.D0*D2DBDD*(PSI+DEL*DPDD)
     3                             + D3DBDD*DEL*PSI)
        ENDIF
        IF (LPHDDT) THEN
          D3DBDT = -2.D0*WB(I) * (BM1*(BM2*THETA*DBM3*DDDD*DDDD
     1                         +2.D0*DBM2*DDDD*DTHDD+THETA*DBM2*D2DDD)
     2                            + DBM1*D2THDD)
          D3PDDT = -4*IWCC(I)*IWDD(I)*TM1*PSI * (2*IWCC(I)*DM1**2-1.D0)
          PHIDDT = PHIDDT + WN(I) * (DB*(2.D0*D2PDTD+DEL*D3PDDT)
     1                             + DDBDT*(2.D0*DPDD+DEL*D2PDD)
     2                             + 2.D0*(DDBDD*(DPDT+DEL*D2PDTD)
     3                                   + D2DBTD*(PSI+DEL*DPDD))
     4                             + D2DBDD*DEL*DPDT + D3DBDT*DEL*PSI)
        ENDIF
        IF (LPHDTT) THEN
          D3DBTD = 2.D0*WB(I)*BM1 * (4.D0*THETA*DTHDD*DBM2
     1                             + 2.D0*THETA*THETA*BM2*DBM3*DDDD
     2                             + DBM2*DDDD)
          D3PDTT = -4*IWCC(I)*IWDD(I)*DM1*PSI * (2*IWDD(I)*TM1**2-1.D0)
          PHIDTT = PHIDTT + WN(I) * (DB*(D2PDT+DEL*D3PDTT)
     1                             + 2.D0*DDBDT*(DPDT+DEL*D2PDTD)
     2                             + DDBDD*DEL*D2PDT 
     3                             + D2DBDT*(PSI+DEL*DPDD)
     4                             + 2.D0*D2DBTD*DEL*DPDT 
     5                             + D3DBTD*DEL*PSI)
        ENDIF
C
  156 CONTINUE
C
      RETURN
      END
