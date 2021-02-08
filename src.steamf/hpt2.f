      FUNCTION HPT2(P,T)                                                 HPT2        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  27
C  HPT2   ENTHALPY - SUB REGION 2    H = F(P,T)   A 2ND LEVEL ROUTINE    HPT2        3                
C         ENTRY = H2E                                                    HPT2        4                
C         CALLS = COMT2                                                  HPT2        5                
C                                                                        HPT2        6                
C                                                                        HPT2        7                
C                                                                        HPT2        8                
      COMMON /CONST2/ BB00, BB01, BB02, BB03, BB04, BB05, BB11, BB12,    HPT2        9                
     1    BB21, BB22, BB23, BB31, BB32, BB41, BB42, BB51, BB52, BB53,    HPT2       10                
     2    BB61, BB62, BB71, BB72, BB81, BB82,  B00,  B61,  B71,  B81,    HPT2       11                
     3     B82, BB90, BB91, BB92, BB93, BB94, BB95, BB96                 HPT2       12                
C                                                                        HPT2       13                
C                                                                        HPT2       14                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  HPT2       15                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  HPT2       16                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  HPT2       17                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          HPT2       18                
C                                                                        HPT2       19                
C                                                                        HPT2       20                
      COMMON /STM2/ THM1  ,THETA ,TH2   ,TH3   ,TH4   ,X1    ,X2    ,    HPT2       21                
     1 X3   ,X4    ,X6    ,X8    ,X10   ,X11   ,X12   ,X13   ,X14   ,    HPT2       22                
     2 X17  ,X18   ,X19   ,X24   ,X25   ,X27   ,X28   ,X32   ,BETA  ,    HPT2       23                
     3 BETA2,BETA3 ,BETA4 ,BETA5 ,BETA6 ,BETA7 ,D4    ,T4    ,D3    ,    HPT2       24                
     4 T3   ,D2    ,T2    ,BETAL ,BOBL  ,BOBLP ,FB    ,BB61F ,BB71F ,    HPT2       25                
     5 BB81F                                                             HPT2       26                
C                                                                        HPT2       27                
      DATA    BB04T2 / -1.309542339D+0 /, BB05T3 / +2.569554617D-1 /,    STEAMV1M2 - 2/96             
     1        B00T2  / +1.526666667D+0 /,  B00T3 / +2.290000000D+0 /,    STEAMV1M2 - 2/96             
     2        B00T4  / +3.053333333D+0 /,  B00T5 / +3.816666667D+0 /,    STEAMV1M2 - 2/96             
     3        B00T6  / +4.580000000D+0 /                                 STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE BB04T2,BB05T3,B00T2,B00T3,B00T4,B00T5,B00T6                   STEAMV1M2 - 3/96
C                                                                        HPT2       32                
C                                                                        HPT2       33                
      CALL COMT2 (P,T)                                                   HPT2       34                
C                                                                        HPT2       35                
      ENTRY H2E(P,T)                                                     STEAMWV1M0(LIB)-NOV. 1,90  28
C                                                                        HPT2       37                
C     THIS ENTRY CAN BE USED ONLY IF COMT2 WAS LAST CALLED WITH THE      HPT2       38                
C     VALUES OF P AND T THAT ARE TO BE ASSUMED HERE.                     HPT2       39                
C                                                                        HPT2       40                
 1000 BOTH = B00*THETA                                                   HPT2       41                
      H0 = BB00*THETA + BB01 - TH2*(BB03 + THETA*(BB04T2 + BB05T3*THETA) HPT2       42                
     1 )                                                                 HPT2       43                
      H1 = BETA*(BB11*X13*(1.0D0 + 13.0D0*BOTH) + BB12*X3*(1.0D0         HPT2       44                
     1 + 3.0*BOTH) + BETA*(BB21*X18*(1.0 + 18.0*BOTH) + BB22*X2*(1.0     HPT2       45                
     2 + 2.0*BOTH) + BB23*X1*(1.0 + BOTH) + BETA*(BB31*X18*(1.0          HPT2       46                
     3 + 18.0*BOTH) + BB32*X10*(1.0 + 10.0*BOTH) + BETA*(BB41*X25*(1.0   HPT2       47                
     4 + 25.0*BOTH) + BB42*X14*(1.0 + 14.0*BOTH) + BETA*(BB51*X32*(1.0   HPT2       48                
     5 + 32.0*BOTH) + BB52*X28*(1.0 + 28.0*BOTH) + BB53*X24*(1.0         HPT2       49                
     6 + 24.0*BOTH))))))                                                 HPT2       50                
      H2A = 1.0 - BOTH*T2                                                HPT2       51                
      H2 = X11*(BB61F*(H2A + 12.0*BOTH) + BB62*(H2A + 11.0*BOTH))/D2     HPT2       52                
      H3A = 1.0 - BOTH*T3                                                HPT2       53                
      H3 = X18*(BB71F*(H3A + 24.0*BOTH) + BB72*(H3A + 18.0*BOTH))/D3     HPT2       54                
      H4A = 1.0 - BOTH*T4                                                HPT2       55                
      H4 = X14*(BB81F*(H4A + 24.0*BOTH) + BB82*(H4A + 14.0*BOTH))/D4     HPT2       56                
      H9 = BETA*BOBLP*((1.0 + THETA*FB)*BB90                             HPT2       57                
     1 + X1*((1.0 + THETA*(FB + B00  ))*BB91                             HPT2       58                
     2 + X1*((1.0 + THETA*(FB + B00T2))*BB92                             HPT2       59                
     3 + X1*((1.0 + THETA*(FB + B00T3))*BB93                             HPT2       60                
     4 + X1*((1.0 + THETA*(FB + B00T4))*BB94                             HPT2       61                
     5 + X1*((1.0 + THETA*(FB + B00T5))*BB95                             HPT2       62                
     6 + X1*((1.0 + THETA*(FB + B00T6))*BB96)))))))                      HPT2       63                
      HPT2 = (H0 - H1 - H2 - H3 - H4 + H9 + ALPHA0)*PV010                HPT2       64                
 2000 RETURN                                                             HPT2       65                
      END                                                                HPT2       66                
