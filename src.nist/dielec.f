C
      SUBROUTINE DIELEC(TK, RHO, DCON, DEPSDT, DEPSDR, 
     >                  D2EDTT, D2EDRR, D2EDTR, MODE, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** DIELEC ***                                   C
C THIS ROUTINE COMPUTES THE STATIC DIELECTRIC CONSTANT                 C
C OF WATER AT A GIVEN TEMPERATURE AND REDUCED DENSITY                  C
C ACCORDING TO THE EQUATION OF FERNANDEZ ET AL                         C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    I   TEMPERATURE IN KELVINS                              C
C RHO     R    I   DENSITY IN KG/M3                                    C
C DCON    R    O   STATIC DIELECTRIC CONSTANT                          C
C DEPSDT  R    O   DERIVATIVE OF DCON WRT TK AT CONSTANT RHO           C
C DEPSDR  R    O   DERIVATIVE OF DCON WRT RHO AT CONSTANT TK           C 
C D2EDTT  R    O   2ND DERIVATIVE OF DCON WRT TK AT CONSTANT RHO       C
C D2EDRR  R    O   2ND DERIVATIVE OF DCON WRT RHO AT CONSTANT TK       C
C D2EDTR  R    O   MIXED 2ND DERIVATIVE OF DCON WRT TO TK, RHO         C
C MODE    I    I   MODE: 0 = CALC DIELECTRIC CONSTANT ONLY             C
C                        1 = CALC DEPSDT AND DEPSDR ALSO               C
C                        2 = CALC 2ND DERIVATIVES ALSO                 C
C IERR    I    O   RETURN FLAG: 0 = SUCCESSFUL CALCULATION             C
C                               1 = T TOO LOW (BELOW 228), DCON        C
C                                   RETURNED AS ZERO                   C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 27NOV95 - INITIAL IMPLEMENTATION BY AHH                              C
C 07DEC95 - COEFFICIENTS REVISED FROM NEW FIT                          C
C 15DEC95 - AHH: DON'T CALCULATE IF TEMPERATURE TOO LOW                C
C 15FEB96 - AHH: PUT IN CALCULATION OF FIRST DERIVATIVES               C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 09JAN97 - AHH: PUT IN CALCULATION OF SECOND DERIVATIVES              C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'wconst.cmn'
      DIMENSION AN(12), II(11), AJJ(11)
      PARAMETER (WALPHA=1.636D-40, WMU=6.138D-30, BOLTZ=1.380658D-23,
     >           AV=6.0221367D23)
      DATA AN /.978224486826D0, -.957771379375D0, .237511794148D0,
     >         .714692244396D0, -.298217036956D0, -.108863472196D0,
     >         .949327488264D-1, -.980469816509D-2, .165167634970D-4,
     >         .937359795772D-4, -.123179218720D-9, .196096504426D-2/
      DATA II /1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 10/
      DATA AJJ /0.25D0, 1.D0, 2.5D0, 1.5D0, 1.5D0, 2.5D0, 2.D0,
     >          2.D0, 5.D0, 0.5D0, 10.D0/
C
      IERR = 0
      DEPSDT = 0.D0
      DEPSDR = 0.D0
      D2EDTT = 0.D0
      D2EDRR = 0.D0
      D2EDTR = 0.D0
C
C CHECK TO SEE IF TEMPERATURE IS TOO LOW TO CALCULATE
C
      IF (TK .LE. 228.D0) THEN
        IERR = 1
        DCON = 0.D0
        RETURN
      ENDIF
C
C FIGURE REDUCED DENSITY AND INVERSE TEMPERATURE VARIABLES
C
      PI = DACOS(-1.D0)
      EPS0 = 1.D0 / (4.D-7*PI*(299792458.D0)**2)
      RHOR = RHO / RHOCW
      TINVR = TCW / TK
C
C FIGURE G-FACTOR
C
      GFACT = 1.D0
      DO 100 I=1,11
        GFACT = GFACT + AN(I)*RHOR**II(I)*TINVR**AJJ(I)
  100 CONTINUE
      TTHING = (TK/228.D0 - 1.D0)
      GFACT = GFACT + AN(12)*RHOR / TTHING**1.2D0
C
C CONVERT TO MOLAR DENSITY (G-MOLES/M3) USING MOLAR MASS
C
      RMOL = 1.D3 * RHO / WMMASS
C
C FIGURE "A" AND "B" QUANTITIES
C
      A = AV*WMU*WMU*RMOL*GFACT / (EPS0*BOLTZ*TK)
      B = AV*WALPHA*RMOL / (3.D0*EPS0)
C
C COMPUTE DIELECTRIC CONSTANT
C
      C = 9.D0 + 2.D0*A + 18.D0*B + A*A + 1.D1*A*B + 9.D0*B*B
      CROOT = DSQRT(C)
      THING1 = 1.D0 + A + 5.D0*B
      THING2 = 1.D0 / (4.D0-4.D0*B)
      DCON = (THING1 + CROOT) * THING2
C
C COMPUTE FIRST DERIVATIVES IF REQUESTED
C
      IF (MODE .GT. 0) THEN
        GRSUM = 0.D0
        GTSUM = 0.D0
        DO 200 I=1,11
          RTERM = AN(I)*RHOR**II(I)
          TTERM = TINVR**AJJ(I)
          GRSUM = GRSUM + II(I)*RTERM*TTERM
          GTSUM = GTSUM + AJJ(I)*RTERM*TTERM
  200   CONTINUE
        DGDR = GRSUM/RHO + AN(12)/RHOCW/TTHING**1.2
        DGDT = -GTSUM/TK - (1.2D0/228.D0)*AN(12)*RHOR/TTHING**2.2
        DADR = A/RHO + A*DGDR/GFACT
        DBDR = B / RHO
        DADT = -A/TK + A*DGDT/GFACT
        DCDR = 2.D0*DADR*THING1 + 2.D0*DBDR*(9.D0+5.D0*A+9.D0*B)
        DCDT = 2.D0*DADT * THING1
        DEPSDT = THING2 * (DADT + 0.5*DCDT/CROOT) 
        DEPSDR = (DADR + 5.D0*DBDR + 0.5*DCDR/CROOT + 4.D0*DCON*DBDR)
     1           * THING2
      ENDIF
C
C COMPUTE SECOND DERIVATIVES IF REQUESTED
C
      IF (MODE .GT. 1) THEN
        GRSUM = 0.D0
        GTSUM = 0.D0
        GTRSUM = 0.D0
        DO 300 I=1,11
          RTERM = AN(I)*RHOR**II(I)
          TTERM = TINVR**AJJ(I)
          GRSUM = GRSUM + (II(I)*II(I)-II(I))*RTERM*TTERM
          GTSUM = GTSUM + (AJJ(I)*AJJ(I)+AJJ(I))*RTERM*TTERM
          GTRSUM = GTRSUM + II(I)*AJJ(I)*RTERM*TTERM
  300   CONTINUE
        D2GDRR = GRSUM / (RHO*RHO)
        D2GDTT = GTSUM/(TK*TK) + (2.64D0/(228.D0**2)) *
     1                           AN(12)*RHOR/TTHING**3.2
        D2GDTR = -GTRSUM/(RHO*TK) - 
     1            1.2D0*AN(12)/(228.D0*RHOCW*TTHING**2.2)
        D2ADRR = 2.D0*A*DGDR/(RHO*GFACT) + A*D2GDRR/GFACT
        D2ADTT = 2.D0*A/(TK*TK) - 2.D0*A*DGDT/(GFACT*TK) +
     1           A*D2GDTT/GFACT
        D2ADTR = A*DGDT/(RHO*GFACT) - A/(RHO*TK) + A*D2GDTR/GFACT -
     1           A*DGDR/(GFACT*TK)
        D2CDRR = 2.D0 * (D2ADRR*THING1 + DADR*DADR +
     1                   9.D0*DBDR*DBDR + 10.D0*DADR*DBDR)
        D2CDTT = 2.D0 * (D2ADTT*THING1 + DADT*DADT)
        D2CDTR = 2.D0 * (D2ADTR*THING1 + DADT*(DADR+5.D0*DBDR))
        D2EDTT = THING2 * (D2ADTT + 0.5D0*(D2CDTT-0.5D0*DCDT*DCDT/C)
     1                                   /CROOT)
        D2EDRR = THING2 * (D2ADRR + 8.D0*DEPSDR*DBDR +
     1                     0.5D0*(D2CDRR-0.5*DCDR*DCDR/C)/CROOT)
        D2EDTR = THING2 * (D2ADTR + 4.D0*DBDR*DEPSDT +
     1                     0.5D0*(D2CDTR-0.5D0*DCDT*DCDR/C)/CROOT)
      ENDIF
      RETURN
      END
