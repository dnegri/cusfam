      FUNCTION VPT1(P,T)                                                 VPT1        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  69
C  VPT1   SPECIFIC VOLUME - SUB REGION 1     A 2ND LEVEL SUBROUTINE      VPT1        3                
C         S = F(P,T)                                                     VPT1        4                
C         ENTRIES = V1E                                                  VPT1        5                
C         CALLS = COMT1                                                  VPT1        6                
C                                                                        VPT1        7                
C                                                                        VPT1        8                
C                                                                        VPT1        9                
      COMMON /CONST1/AA0, AA1, AA2, AA3, AA4, AA5, AA6, AA7, AA8, AA9,   VPT1       10                
     1    AA10,AA11,AA12,AA13,AA14,AA15,AA16,AA17,AA18,AA19,AA20,AA21,   VPT1       11                
     2    AA22,  A1,  A2,  A3,  A4,  A5,  A6,  A7,  A8,  A9, A10, A11,   VPT1       12                
     3     A12                                                           VPT1       13                
C                                                                        VPT1       14                
C                                                                        VPT1       15                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  VPT1       16                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  VPT1       17                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  VPT1       18                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          VPT1       19                
C                                                                        VPT1       20                
C                                                                        VPT1       21                
      COMMON /STM1/ THETA,  TH2,  TH4,  TH6,  TH7, TH10, TH11, TH16,     VPT1       22                
     1   TH17, TH18, TH19, TH20, TH21, BETA,BETA2,BETA3,BETA4, Y,YP,     VPT1       23                
     2  Z, ZP,  UA9, UA10,   UB,  UB2,   UC,  U3T,  UD4,   UP            VPT1       24                
C                                                                        VPT1       25                
      DATA    AA18T2 / +4.348040700D-8 /, AA19T3 / +3.317131494D-9 /,    STEAMV1M2 - 2/96             
     1        AA21T3 / +3.924357216D-5 /, AA22T4 / +2.419050535D-13/     STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE AA18T2,AA19T3,AA21T3,AA22T4                                   STEAMV1M2 - 3/96
C                                                                        VPT1       28                
      CALL COMT1 (P,T)                                                   VPT1       29                
C                                                                        VPT1       30                
      ENTRY V1E(P,T)                                                     STEAMWV1M0(LIB)-NOV. 1,90  70
C                                                                        VPT1       32                
C     THIS ENTRY TO BE USED ONLY IF COMT1 WAS LAST CALLED WITH THE       VPT1       33                
C     VALUES OF P AND T THAT ARE TO BE ASSUMED HERE.                     VPT1       34                
C                                                                        VPT1       35                
 1000 V4 = BETA2*(AA21T3*(A12 - THETA) + AA22T4*BETA/TH20)               VPT1       36                
      V3 = AA20*TH18*(A9 + TH2)*(A11 - 3.0/UD4)                          VPT1       37                
      V2 = (AA17 + BETA*(AA18T2 + AA19T3*BETA))/UC                       VPT1       38                
      V1 = AA12 + THETA*(AA13 + THETA*AA14) + AA15*UA10 + AA16/UB        VPT1       39                
      VPT1 = (AA11*A5*ZP + V1 - V2 - V3 + V4)*VCA                        VPT1       40                
 2000 RETURN                                                             VPT1       41                
      END                                                                VPT1       42                
