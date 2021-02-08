      FUNCTION VPT2(P,T)                                                 VPT2        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  71
C  VPT2   SPECIFIC VOLUME - SUB REGION 2     A 2ND LEVEL SUBROUTINE      VPT2        3                
C         S = F(P,T)                                                     VPT2        4                
C         ENTRIES = V2E                                                  VPT2        5                
C         CALLS = COMT2                                                  VPT2        6                
C                                                                        VPT2        7                
C                                                                        VPT2        8                
C                                                                        VPT2        9                
      COMMON /CONST2/ BB00, BB01, BB02, BB03, BB04, BB05, BB11, BB12,    VPT2       10                
     1    BB21, BB22, BB23, BB31, BB32, BB41, BB42, BB51, BB52, BB53,    VPT2       11                
     2    BB61, BB62, BB71, BB72, BB81, BB82,  B00,  B61,  B71,  B81,    VPT2       12                
     3     B82, BB90, BB91, BB92, BB93, BB94, BB95, BB96                 VPT2       13                
C                                                                        VPT2       14                
C                                                                        VPT2       15                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  VPT2       16                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  VPT2       17                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  VPT2       18                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          VPT2       19                
C                                                                        VPT2       20                
C                                                                        VPT2       21                
      COMMON /STM2/ THM1  ,THETA ,TH2   ,TH3   ,TH4   ,X1    ,X2    ,    VPT2       22                
     1 X3   ,X4    ,X6    ,X8    ,X10   ,X11   ,X12   ,X13   ,X14   ,    VPT2       23                
     2 X17  ,X18   ,X19   ,X24   ,X25   ,X27   ,X28   ,X32   ,BETA  ,    VPT2       24                
     3 BETA2,BETA3 ,BETA4 ,BETA5 ,BETA6 ,BETA7 ,D4    ,T4    ,D3    ,    VPT2       25                
     4 T3   ,D2    ,T2    ,BETAL ,BOBL  ,BOBLP ,FB    ,BB61F ,BB71F ,    VPT2       26                
     5 BB81F                                                             VPT2       27                
C                                                                        VPT2       28                
      CALL COMT2 (P,T)                                                   VPT2       29                
C                                                                        VPT2       30                
      ENTRY V2E(P,T)                                                     STEAMWV1M0(LIB)-NOV. 1,90  72
C                                                                        VPT2       32                
C     THIS ENTRY CAN BE USED ONLY IF COMT2 WAS LAST CALLED WITH THE      VPT2       33                
C     VALUES OF P AND T THAT ARE TO BE ASSUMED HERE.                     VPT2       34                
C                                                                        VPT2       35                
 1000 V1 = X3*(BB11*X10 + BB12) + BETA*(2.0*X1*(BB21*X17                 VPT2       36                
     1 + BB22*X1 + BB23) + BETA*(3.0*X10*(BB31*X8 + BB32)                VPT2       37                
     2 + BETA*(4.0*X14*(BB41*X11 + BB42)                                 VPT2       38                
     3 + BETA*(5.0*X24*(BB51*X8 + BB52*X4 + BB53)))))                    VPT2       39                
C                                                                        VPT2       40                
C     V4, V3, AND V2 MUST BE EXPRESSED IN THIS WAY TO AVOID              VPT2       41                
C         OVERFLOWS AT LOW PRESSURES.                                    VPT2       42                
C                                                                        VPT2       43                
      V2 = 4.0*X11*(BB61F + BB62)/(D2*BETA5*D2)                          VPT2       44                
      V3 = 5.0*X18*(BB71F + BB72)/(D3*BETA6*D3)                          VPT2       45                
      V4 = 6.0*X14*(BB81F + BB82)/(D4*BETA7*D4)                          VPT2       46                
      V9 = 11.0*(BOBLP*(BB90 + X1*(BB91 + X1*(BB92 + X1*(BB93            VPT2       47                
     1 + X1*(BB94 + X1*(BB95 + X1*BB96)))))))                            VPT2       48                
      VPT2 = (AI1*THETA/BETA - V1 - V2 - V3 - V4 + V9)*VCA               VPT2       49                
 2000 RETURN                                                             VPT2       50                
      END                                                                VPT2       51                
