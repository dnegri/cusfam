      FUNCTION CPPT1(P,T)                                                CPPT1       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  16
C         SPECIFIC HEAT - SUB REGION 1                                   CPPT1       3                
C         A 2ND LEVEL SUBROUTINE                                         CPPT1       4                
C         WITH ENTRY CP1E                                                CPPT1       5                
C                                                                        CPPT1       6                
C                                                                        CPPT1       7                
C     FIRST ARGUMENT IS PRESSURE                                         CPPT1       8                
C     SECOND ARGRMENT IS TEMPERATURE                                     CPPT1       9                
C     RETURNS WITH SPECIFIC HEAT                                         CPPT1      10                
C                                                                        CPPT1      11                
      COMMON /CONST1/AA0, AA1, AA2, AA3, AA4, AA5, AA6, AA7, AA8, AA9,   CPPT1      12                
     1    AA10,AA11,AA12,AA13,AA14,AA15,AA16,AA17,AA18,AA19,AA20,AA21,   CPPT1      13                
     2    AA22,  A1,  A2,  A3,  A4,  A5,  A6,  A7,  A8,  A9, A10, A11,   CPPT1      14                
     3     A12                                                           CPPT1      15                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  CPPT1      16                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  CPPT1      17                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  CPPT1      18                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          CPPT1      19                
      COMMON /STM1/ THETA,  TH2,  TH4,  TH6,  TH7, TH10, TH11, TH16,     CPPT1      20                
     1   TH17, TH18, TH19, TH20, TH21, BETA,BETA2,BETA3,BETA4, Y,YP,     CPPT1      21                
     2  Z, ZP,  UA9, UA10,   UB,  UB2,   UC,  U3T,  UD4,   UP            CPPT1      22                
C                                                                        CPPT1      23                
      DATA AA3T2, AA4T6 / +7.882573574D4, -4.039966643D5 /               STEAMV1M2 - 2/96
      DATA AA5T12, AA6T20 / +1.188285723D6, -2.187823548D6 /             STEAMV1M2 - 2/96
      DATA AA7T30, AA8T42 / +2.577252500D6, -1.894690872D6 /             STEAMV1M2 - 2/96
      DATA AA9T56, AA1072 / +7.941577986D5, -1.452435201D5 /             STEAMV1M2 - 2/96
      DATA AA14T2, AA1590 / +4.568558108D-2, 2.179482303D4 /             STEAMV1M2 - 2/96
      DATA AA22B / +2.540003062D-11 /                                    STEAMV1M2 - 2/96
      DATA A1T2, A2T6 / +1.687675081D0, +3.217297297D-3 /                STEAMV1M2 - 2/96
      DATA A2T42, A9T306 / +2.252108108D-2, +4.341528000D1 /             STEAMV1M2 - 2/96
      DATA A4T2, A5T2 / +1.468455698D-1, +9.951717740D-2 /               STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 2/96
      SAVE AA3T2,AA4T6,AA5T12,AA6T20,AA7T30,AA8T42,AA9T56,AA1072         STEAMV1M2 - 3/96
      SAVE AA14T2,AA1590,AA22B,A1T2,A2T6,A2T42,A9T306,A4T2,A5T2          STEAMV1M2 - 3/96
C                                                                        CPPT1      33                
      CALL COMT1(P,T)                                                    CPPT1      34                
C                                                                        CPPT1      35                
      ENTRY CP1E(P,T)                                                    STEAMWV1M0(LIB)-NOV. 1,90  17
C                                                                        CPPT1      37                
C     THIS ENTRY CAN BE USED ONLY IF COMT1 WAS LAST CALLED WITH THE      CPPT1      38                
C     VALUES OF P AND T THAT ARE TO BE ASSUMED HERE.                     CPPT1      39                
C                                                                        CPPT1      40                
 1000 UA = A6 - THETA                                                    CPPT1      41                
      UA8 = UA**8                                                        CPPT1      42                
      WW = A3*Y*Y + A5T2*BETA - A4T2*THETA                               CPPT1      43                
      W = DSQRT(WW)                                                      STEAMWV1M0(LIB)-NOV. 1,90  18
      Z12 = Z**(12.0D0/17.0D0)                                           STEAMV1M2 - 2/96                             
      Z22 = Z**(-22.0D0/17.0D0)                                          STEAMV1M2 - 2/96                             
      DYDTH = -A1T2*THETA + A2T6/TH7                                     CPPT1      47                
      WA = (A3*Y*DYDTH - A4)/W                                           CPPT1      48                
      DYDTH2 = -A1T2 - A2T42/(TH7*THETA)                                 CPPT1      49                
      DZDTH = DYDTH + WA                                                 CPPT1      50                
      DZDTH2 = DYDTH2 +(A3/W)*(DYDTH*DYDTH + Y*DYDTH2) - WA*WA/W         CPPT1      51                
      CP1 = AA0 - THETA*(AA3T2 + THETA*(AA4T6 + THETA*(AA5T12 + THETA*(  CPPT1      52                
     1 AA6T20 + THETA*(AA7T30 + THETA*(AA8T42 + THETA*(AA9T56 + THETA*   CPPT1      53                
     2 AA1072)))))))                                                     CPPT1      54                
      CP2 = AA11*THETA*(12.0D0*(Z/29.0D0 - Y/12.0D0)*(ZP*DZDTH2          STEAMV1M2 - 2/96                             
     1  - 5.0D0*Z22*DZDTH*DZDTH/17.0D0) + 24.0D0*(DZDTH/29.0D0           STEAMV1M2 - 2/96                             
     2 - DYDTH/12.0D0)*ZP*DZDTH + 17.0D0*(DZDTH2/29.0D0                  STEAMV1M2 - 2/96                             
     3 - DYDTH2/12.0D0)*Z12)                                             STEAMV1M2 - 2/96                             
      CP3 = THETA*BETA*(AA14T2 + AA1590*UA8                              CPPT1      59                
     1 + TH17*AA16*(722.0D0*TH19/UB - 342.0D0)/UB2)                      STEAMV1M2 - 2/96                             
      CP4 = TH10*(242.0D0*TH11/UC - 110.0D0)*U3T                         STEAMV1M2 - 2/96                             
      CP5 = AA20*TH17*(A9T306 + 380.0D0*TH2)*UP                          STEAMV1M2 - 2/96                             
      CP6 = AA22B*BETA4/TH21                                             CPPT1      63                
      CPPT1 = (CP1 - CP2 - CP3 + CP4 + CP5 - CP6)*PVOT                   CPPT1      64                
 2000 RETURN                                                             CPPT1      65                
      END                                                                CPPT1      66                
