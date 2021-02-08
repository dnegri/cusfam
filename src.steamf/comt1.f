      SUBROUTINE COMT1 (PE,TE)                                           COMT1       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90   3
C  COMT1  COMMON TERMS  - SUB REGION 1                                   COMT1       3                
C         A 1ST LEVEL SUBROUTINE                                         COMT1       4                
C                                                                        COMT1       5                
C                                                                        COMT1       6                
C     FIRST ARGUMENT IS PRESSURE                                         COMT1       7                
C     SECOND ARGUMENT IS TEMPERATURE                                     COMT1       8                
C                                                                        COMT1       9                
C                                                                        COMT1      10                
      COMMON /CONST1/AA0, AA1, AA2, AA3, AA4, AA5, AA6, AA7, AA8, AA9,   COMT1      11                
     1    AA10,AA11,AA12,AA13,AA14,AA15,AA16,AA17,AA18,AA19,AA20,AA21,   COMT1      12                
     2    AA22,  A1,  A2,  A3,  A4,  A5,  A6,  A7,  A8,  A9, A10, A11,   COMT1      13                
     3     A12                                                           COMT1      14                
C                                                                        COMT1      15                
C                                                                        COMT1      16                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  COMT1      17                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  COMT1      18                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  COMT1      19                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          COMT1      20                
C                                                                        COMT1      21                
      COMMON /CURENT / P, T, V, H, S                                     COMT1      22                
C                                                                        COMT1      23                
C                                                                        COMT1      24                
      COMMON /STM1/ THETA,  TH2,  TH4,  TH6,  TH7, TH10, TH11, TH16,     COMT1      25                
     1   TH17, TH18, TH19, TH20, TH21, BETA,BETA2,BETA3,BETA4, Y,YP,     COMT1      26                
     2  Z, ZP,  UA9, UA10,   UB,  UB2,   UC,  U3T,  UD4,   UP            COMT1      27                
C                                                                        COMT1      28                
      character cccc*6
      data cccc/6H COMT1/
      P = PE                                                             COMT1      29                
      T = TE                                                             COMT1      30                
      IF(P .LT. PMIN .OR. P .GT. PMAX) CALL STER(cccc, 12, P, T)         COMT1      31                
      IF(T .LT. TMIN .OR. T .GT.T1MAX) CALL STER(cccc, 12, P, T)         COMT1      32                
      BETA = P/PCA                                                       COMT1      33                
      THETA = (T + TZA)/TCA                                              COMT1      34                
 1000 TH2 = THETA * THETA                                                COMT1      35                
      TH4 = TH2 * TH2                                                    COMT1      36                
      TH6 = TH2 * TH4                                                    COMT1      37                
      TH7 = TH6 * THETA                                                  COMT1      38                
      TH10 = TH4 * TH6                                                   COMT1      39                
      TH11 = TH4 * TH7                                                   COMT1      40                
      TH16 = TH6 * TH10                                                  COMT1      41                
      TH17 = TH7 * TH10                                                  COMT1      42                
      TH18 = TH2 * TH16                                                  COMT1      43                
      TH19 = TH2 * TH17                                                  COMT1      44                
      TH20 = TH4 * TH16                                                  COMT1      45                
      TH21 = TH2 * TH19                                                  COMT1      46                
      BETA2 = BETA * BETA                                                COMT1      47                
      BETA3 = BETA * BETA2                                               COMT1      48                
      BETA4 = BETA * BETA3                                               COMT1      49                
      UA = A6 - THETA                                                    COMT1      50                
      UA9 = UA**9                                                        COMT1      51                
 3000 UA10 = UA * UA9                                                    COMT1      52                
      UB = A7 + TH19                                                     COMT1      53                
      UB2 = UB*UB                                                        COMT1      54                
      UC = A8 + TH11                                                     COMT1      55                
      U3T = BETA * (AA17 + BETA * (AA18 + AA19*BETA))/UC/UC              COMT1      56                
      UD = A10 + BETA                                                    COMT1      57                
      UD3 = UD*UD*UD                                                     COMT1      58                
      UD4 = UD*UD3                                                       COMT1      59                
      UP = 1.0/UD3 + A11*BETA                                            COMT1      60                
      Y = 1.0 - A1*TH2 - A2/TH6                                          COMT1      61                
      Z = Y +  DSQRT(A3*Y*Y + 2.0 *(A5*BETA - A4*THETA))                 STEAMWV1M0(LIB)-NOV. 1,90   4
      YP = 6.0*A2/TH7 - 2.0*THETA*A1                                     COMT1      63                
      ZP = Z**(-5.0/17.0)                                                COMT1      64                
 6000 RETURN                                                             COMT1      65                
      END                                                                COMT1      66                
