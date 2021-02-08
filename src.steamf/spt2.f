      FUNCTION SPT2(P,T)                                                 SPT2        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  47
C  SPT2   ENTROPY = F(P,T) SUB REGION 2           A 2ND LEVEL SUBROUTINE SPT2        3                
C         ENTRIES = S2E                                                  SPT2        4                
C         CALLS = COMT2                                                  SPT2        5                
C                                                                        SPT2        6                
C                                                                        SPT2        7                
C                                                                        SPT2        8                
      COMMON /CONST2/ BB00, BB01, BB02, BB03, BB04, BB05, BB11, BB12,    SPT2        9                
     1    BB21, BB22, BB23, BB31, BB32, BB41, BB42, BB51, BB52, BB53,    SPT2       10                
     2    BB61, BB62, BB71, BB72, BB81, BB82,  B00,  B61,  B71,  B81,    SPT2       11                
     3     B82, BB90, BB91, BB92, BB93, BB94, BB95, BB96                 SPT2       12                
C                                                                        SPT2       13                
C                                                                        SPT2       14                
C                                                                        SPT2       15                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  SPT2       16                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  SPT2       17                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  SPT2       18                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          SPT2       19                
C                                                                        SPT2       20                
C                                                                        SPT2       21                
C                                                                        SPT2       22                
      COMMON /STM2/ THM1  ,THETA ,TH2   ,TH3   ,TH4   ,X1    ,X2    ,    SPT2       23                
     1 X3   ,X4    ,X6    ,X8    ,X10   ,X11   ,X12   ,X13   ,X14   ,    SPT2       24                
     2 X17  ,X18   ,X19   ,X24   ,X25   ,X27   ,X28   ,X32   ,BETA  ,    SPT2       25                
     3 BETA2,BETA3 ,BETA4 ,BETA5 ,BETA6 ,BETA7 ,D4    ,T4    ,D3    ,    SPT2       26                
     4 T3   ,D2    ,T2    ,BETAL ,BOBL  ,BOBLP ,FB    ,BB61F ,BB71F ,    SPT2       27                
     5 BB81F                                                             SPT2       28                
C                                                                        SPT2       29                
      DATA    BB03T2 / +8.661325668D-1 /, BB04T3 / -1.964313509D+0 /,    STEAMV1M2 - 2/96             
     1        BB05T4 / +3.426072823D-1 /, BB1113 / +8.671488693D-1 /,    STEAMV1M2 - 2/96             
     2        BB12T3 / +4.166951403D+0 /, BB2118 / +1.510218779D+0 /,    STEAMV1M2 - 2/96             
     3        BB22T2 / +5.229341786D-2 /, BB3118 / +8.137654027D+0 /,    STEAMV1M2 - 2/96             
     4        BB3210 / +1.069036614D+0 /, BB4125 / -1.493834177D+1 /,    STEAMV1M2 - 2/96             
     5        BB4214 / -1.238655012D+0 /, BB5132 / +1.906576515D+1 /,    STEAMV1M2 - 2/96             
     6        BB5228 / -1.444604944D+1 /, BB5324 / +4.980050693D+0 /,    STEAMV1M2 - 2/96             
     7        B00T2  / +1.526666667D+0 /, B00T3  / +2.290000000D+0 /,    STEAMV1M2 - 2/96             
     8        B00T4  / +3.053333333D+0 /, B00T5  / +3.816666667D+0 /,    STEAMV1M2 - 2/96             
     9        B00T6  / +4.580000000D+0 /                                 STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE BB03T2,BB04T3,BB05T4,BB1113,BB12T3,BB2118,BB22T2,BB3118       STEAMV1M2 - 3/96
      SAVE BB3210,BB4125,BB4214,BB5132,BB5228,BB5324,B00T2,B00T3         STEAMV1M2 - 3/96
      SAVE B00T4,B00T5,B00T6                                             STEAMV1M2 - 3/96
C                                                                        SPT2       40                
      CALL COMT2 (P,T)                                                   SPT2       41                
C                                                                        SPT2       42                
      ENTRY S2E(P,T)                                                     STEAMWV1M0(LIB)-NOV. 1,90  48
C                                                                        SPT2       44                
C     THIS ENTRY CAN BE USED ONLY IF COMT2 WAS LAST CALLED WITH THE      SPT2       45                
C     VALUES OF P AND T THAT ARE TO BE ASSUMED HERE.                     SPT2       46                
C                                                                        SPT2       47                
 1000 S0 = BB00*DLOG(THETA) - AI1*DLOG(BETA) - BB02                      STEAMWV1M0(LIB)-NOV. 1,90  49
     1 - THETA*(BB03T2 + THETA*(BB04T3 + THETA*BB05T4))                  SPT2       49                
      S1 = BETA*(X3*(BB1113*X10 + BB12T3)                                SPT2       50                
     1 + BETA*(X1 *(BB2118*X17 + BB22T2*X1 + BB23)                       SPT2       51                
     2 + BETA*(X10*(BB3118*X8  + BB3210)                                 SPT2       52                
     3 + BETA*(X14*(BB4125*X11 + BB4214)                                 SPT2       53                
     4 + BETA*(X24*(BB5132*X8  + BB5228*X4 + BB5324))))))                SPT2       54                
      S2 = X11*(BB61F*(12.0 - T2) + BB62*(11.0 - T2))/D2                 SPT2       55                
      S3 = X18*(BB71F*(24.0 - T3) + BB72*(18.0 - T3))/D3                 SPT2       56                
      S4 = X14*(BB81F*(24.0 - T4) + BB82*(14.0 - T4))/D4                 SPT2       57                
      S9 = BETA*BOBLP*(FB*BB90 + X1*((FB + B00)*BB91                     SPT2       58                
     1 + X1*((FB + B00T2)*BB92 + X1*((FB + B00T3)*BB93                   SPT2       59                
     2 + X1*((FB + B00T4)*BB94 + X1*((FB + B00T5)*BB95                   SPT2       60                
     3 + X1*((FB + B00T6)*BB96)))))))                                    SPT2       61                
      SPT2 = (S0 - B00*(S1 + S2 + S3 + S4) + S9 - ALPHA1)*PVOT           SPT2       62                
 2000 RETURN                                                             SPT2       63                
      END                                                                SPT2       64                
