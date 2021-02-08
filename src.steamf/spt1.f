      FUNCTION SPT1(P,T)                                                 SPT1        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  44
C  SPT1   S = F(P,T)     ENTROPY - SUBREGION 1    A 2ND LEVEL ROUTINE    SPT1        3                
C         ENTRY = S1E(DUMMY)                                             SPT1        4                
C                                                                        SPT1        5                
C                                                                        SPT1        6                
C                                                                        SPT1        7                
      COMMON /CONST1/AA0, AA1, AA2, AA3, AA4, AA5, AA6, AA7, AA8, AA9,   SPT1        8                
     1    AA10,AA11,AA12,AA13,AA14,AA15,AA16,AA17,AA18,AA19,AA20,AA21,   SPT1        9                
     2    AA22,  A1,  A2,  A3,  A4,  A5,  A6,  A7,  A8,  A9, A10, A11,   SPT1       10                
     3     A12                                                           SPT1       11                
C                                                                        SPT1       12                
C                                                                        SPT1       13                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  SPT1       14                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  SPT1       15                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  SPT1       16                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          SPT1       17                
C                                                                        SPT1       18                
C                                                                        SPT1       19                
      COMMON /STM1/ THETA,  TH2,  TH4,  TH6,  TH7, TH10, TH11, TH16,     SPT1       20                
     1   TH17, TH18, TH19, TH20, TH21, BETA,BETA2,BETA3,BETA4, Y,YP,     SPT1       21                
     2  Z, ZP,  UA9, UA10,   UB,  UB2,   UC,  U3T,  UD4,   UP            SPT1       22                
C                                                                        SPT1       23                
      DATA A3M1  , A9T18  / +7.200000000D-1, +2.553840000D+0  /          STEAMV1M2 - 2/96             
      DATA AA3T2 , AA4T3  / +7.882573574D+4, -2.019983322D+5  /          STEAMV1M2 - 2/96             
      DATA AA5T4 , AA6T5  / +3.960952411D+5, -5.469558870D+5  /          STEAMV1M2 - 2/96             
      DATA AA7T6 , AA8T7  / +5.154505000D+5, -3.157818119D+5  /          STEAMV1M2 - 2/96             
      DATA AA9T8 , AA10T9 / +1.134511141D+5, -1.815544002D+4  /          STEAMV1M2 - 2/96             
      DATA AA14T2, AA1510 / +4.568558108D-2, +2.421647003D+3  /          STEAMV1M2 - 2/96             
      DATA AA1619, AA2220 / +2.412460567D-9, +1.209525268D-12 /          STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE A3M1,A9T18,AA3T2,AA4T3,AA5T4,AA6T5,AA7T6,AA8T7,AA9T8,AA10T9   STEAMV1M2 - 3/96
      SAVE AA14T2,AA1510,AA1619,AA2220                                   STEAMV1M2 - 3/96
C                                                                        SPT1       31                
      CALL COMT1 (P,T)                                                   SPT1       32                
C                                                                        SPT1       33                
      ENTRY S1E(P,T)                                                     STEAMWV1M0(LIB)-NOV. 1,90  45
C                                                                        SPT1       35                
C     THIS ENTRY TO BE USED ONLY IF COMT1 WAS LAST CALLED WITH THE       SPT1       36                
C     VALUES OF P AND T THAT ARE TO BE ASSUMED HERE.                     SPT1       37                
C                                                                        SPT1       38                
 1000 S5 = BETA3*(AA21 + AA2220*BETA/TH21)                               SPT1       39                
      S4 = AA20*TH17*(A9T18 + 20.0*TH2)*UP                               SPT1       40                
      S3 = 11.0*TH10*U3T                                                 SPT1       41                
      S2 = (-AA13 - AA14T2*THETA + AA1510*UA9 + AA1619*TH18/UB2)*BETA    SPT1       42                
      S1 = AA11*((5.0*Z/12.0 - A3M1*Y)*YP + A4)*ZP                       SPT1       43                
      S0 = AA0* DLOG(THETA) - (AA2 + THETA*(AA3T2 + THETA*(AA4T3         STEAMWV1M0(LIB)-NOV. 1,90  46
     1 + THETA*(AA5T4 + THETA*(AA6T5 + THETA*(AA7T6 + THETA*(AA8T7       SPT1       45                
     2 + THETA*(AA9T8 + THETA*AA10T9))))))))                             SPT1       46                
C                                                                        SPT1       47                
C     CONSTANT ADDED TO GIVE EXACT VALUE AT TRIPLE POINT.                SPT1       48                
C                                                                        SPT1       49                
      SPT = S0 + S1 + S2 - S3 + S4 + S5                                  SPT1       50                
      SPT1 = (SPT - ALPHA1)*PVOT                                         SPT1       51                
 2000 RETURN                                                             SPT1       52                
      END                                                                SPT1       53                
