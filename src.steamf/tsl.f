      FUNCTION TSL(PIN)                                                  TSL         2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  57
C  TSL    TSAT = F(P) SATURATION TEMPERATURE AS A FUNCTION OF PRESSURE.  TSL         3                
C         ENTRIES = TSATP                                                TSL         4                
C         CALLS = PSL, GRS, STER.                                        TSL         5                
C                                                                        TSL         6                
C                                                                        TSL         7                
C                                                                        TSL         8                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  TSL         9                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  TSL        10                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  TSL        11                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          TSL        12                
C                                                                        TSL        13                
      COMMON /NUST/ DPDT                                                 TSL        14                
      DIMENSION B(12)                                                    TSL        15                
      character cccc*6
      data cccc/3hTSL/
      DATA  B / +1.52264682686D+0, -0.682309517937 , +0.164114951728  ,  STEAMV1M2 - 2/96             
     1          -2.02321648831D-3, -1.92391110748D-3,-5.74549418696D-4,  STEAMV1M2 - 2/96             
     2          +6.84115542402D-5, +3.36500068426D-5,-1.23422483951D-5,  STEAMV1M2 - 2/96             
     3          +1.48265501702D-6, -1.02116445578D-6,-4.09080904092D-6 / STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE B                                                             STEAMV1M2 - 3/96
C                                                                        TSL        20                
      ENTRY TSATP(PIN)                                                   STEAMWV1M0(LIB)-NOV. 1,90  58
      I=1                                                                TSL        22                
      TOL=1.0D-4                                                         TSL        23                
      IF(PIN.GT.2700.) TOL=1.0D-9                                        TSL        24                
      IF (PIN.GT.3200.) TOL=1.0D-10                                      TSL        25                
      F=1.                                                               TSL        26                
      IF(PIN.GT.PCA) GO TO 30                                            TSL        27                
      TSL=TC                                                             TSL        28                
      IF(PIN.GE.3208.2347) GO TO 10                                      TSL        29                
      I=-1                                                               TSL        30                
      TX=1.                                                              TSL        31                
      TY=(DLOG(3529.058235/PIN)**0.4-1.48047125)/(-1.089944005)          STEAMWV1M0(LIB)-NOV. 1,90  59
      Y=2.*TY                                                            TSL        33                
      W=B(1)+TY*B(2)                                                     TSL        34                
      DO5N=3,12                                                          TSL        35                
      TZ=Y*TY-TX                                                         TSL        36                
      W=W+TZ*B(N)                                                        TSL        37                
      TX=TY                                                              TSL        38                
      TY=TZ                                                              TSL        39                
  5   CONTINUE                                                           TSL        40                
      TSL=TCA/W-TZA                                                      TSL        41                
  10  TY=.01                                                             TSL        42                
      Y=1.                                                               TSL        43                
      IF(TSL.GT.TC) TSL=TC                                               TSL        44                
  15  PA=PSL1(TSL)                                                       TSL        45                
      DP=PIN-PA                                                          TSL        46                
      PR=DP/PIN                                                          TSL        47                
      TSL=TSL+F*DP/DPDT                                                  TSL        48                
      IF(DABS(PR)-TOL) 35,20,20                                          STEAMWV1M0(LIB)-NOV. 1,90  60
  20  IF(Y.GT.29.) GO TO 30                                              TSL        50                
      Y=Y+1.                                                             TSL        51                
      F=F*0.99                                                           TSL        52                
      IF(TSL-TC) 15,25,25                                                TSL        53                
  25  TY=0.9*TY                                                          TSL        54                
      TSL=TSL-TY                                                         TSL        55                
      GO TO 15                                                           TSL        56                
  30  CALL STER(cccc,I,PIN,0.d0)                                            TSL        57                
  35  RETURN                                                             TSL        58                
C     END OF TSL                                                         TSL        59                
      END                                                                TSL        60                
