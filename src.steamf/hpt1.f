      FUNCTION HPT1(P,T)                                                 HPT1        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  25
C  HPT1  ENTHALPY - SUBREGION 1  H = F(P,T)  A 2ND LEVEL SUBROUTINE      HPT1        3                
C        ENTRY = H1E(DUMMY)                                              HPT1        4                
C        CALLS COMT1(P,T)                                                HPT1        5                
C                                                                        HPT1        6                
C                                                                        HPT1        7                
C                                                                        HPT1        8                
C                                                                        HPT1        9                
      COMMON /CONST1/AA0, AA1, AA2, AA3, AA4, AA5, AA6, AA7, AA8, AA9,   HPT1       10                
     1    AA10,AA11,AA12,AA13,AA14,AA15,AA16,AA17,AA18,AA19,AA20,AA21,   HPT1       11                
     2    AA22,  A1,  A2,  A3,  A4,  A5,  A6,  A7,  A8,  A9, A10, A11,   HPT1       12                
     3     A12                                                           HPT1       13                
C                                                                        HPT1       14                
C                                                                        HPT1       15                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  HPT1       16                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  HPT1       17                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  HPT1       18                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          HPT1       19                
C                                                                        HPT1       20                
C                                                                        HPT1       21                
      COMMON /STM1/ THETA,  TH2,  TH4,  TH6,  TH7, TH10, TH11, TH16,     HPT1       22                
     1   TH17, TH18, TH19, TH20, TH21, BETA,BETA2,BETA3,BETA4, Y,YP,     HPT1       23                
     2  Z, ZP,  UA9, UA10,   UB,  UB2,   UC,  U3T,  UD4,   UP            HPT1       24                
C                                                                        HPT1       25                
C                                                                        HPT1       26                
      DATA   A3M1,  A9T17 / +7.200000000D-1, +2.411960000D00 /           STEAMV1M2 - 2/96             
      DATA  AA4T2,  AA5T3 / -1.346655548D+5, +2.970714308D+5 /           STEAMV1M2 - 2/96             
      DATA  AA6T4,  AA7T5 / -4.375647096D+5, +4.295420834D+5 /           STEAMV1M2 - 2/96             
      DATA  AA8T6,  AA9T7 / -2.706701245D+5, +9.926972482D+4 /           STEAMV1M2 - 2/96             
      DATA AA10T8, AA2221 / -1.613816890D+4, +1.270001531D-12/           STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE A3M1,A9T17,AA4T2,AA5T3,AA6T4,AA7T5,AA8T6,AA9T7,AA10T8,AA2221  STEAMV1M2 - 3/96
C                                                                        HPT1       32                
C                                                                        HPT1       33                
      CALL COMT1 (P,T)                                                   HPT1       34                
C                                                                        HPT1       35                
      ENTRY H1E(P,T)                                                     STEAMWV1M0(LIB)-NOV. 1,90  26
C                                                                        HPT1       37                
C     THIS ENTRY TO BE USED ONLY IF COMT1 WAS LAST CALLED WITH THE       HPT1       38                
C     VALUES OF P AND T THAT ARE TO BE ASSUMED HERE.                     HPT1       39                
C                                                                        HPT1       40                
 1000 H5 =BETA3*(AA21*A12 + AA2221*BETA/TH20)                            HPT1       41                
      H4 = AA20*TH18*(A9T17 + 19.0*TH2)*UP                               HPT1       42                
      H3 = (12.0 *TH11 + A8)*U3T                                         HPT1       43                
      H2 = (AA12 - AA14*TH2 + AA15*(9.0*THETA + A6)*UA9                  HPT1       44                
     1 + AA16*(20.*TH19 + A7)/UB2)*BETA                                  HPT1       45                
      H1 = AA11*(Z*(17.0*(Z/29.0 - Y/12.0) + 5.0*THETA*YP/12.0) +        HPT1       46                
     1 A4*THETA - A3M1*THETA*Y*YP)*ZP                                    HPT1       47                
      H0 = AA0*THETA + AA1 - TH2*(AA3 + THETA*(AA4T2 + THETA*(AA5T3 +    HPT1       48                
     1 THETA*(AA6T4 + THETA*(AA7T5 + THETA*(AA8T6 + THETA*(AA9T7 +       HPT1       49                
     2 THETA*AA10T8)))))))                                               HPT1       50                
C                                                                        HPT1       51                
C     CONSTANT ADDED TO GIVE EXACT VALUE AT TRIPLE POINT.                HPT1       52                
C                                                                        HPT1       53                
      HPT1 = (H0 + H1 + H2 - H3 + H4 + H5 + ALPHA0)*PV010                HPT1       54                
 2000 RETURN                                                             HPT1       55                
      END                                                                HPT1       56                
