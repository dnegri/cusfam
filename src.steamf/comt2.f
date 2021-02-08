      SUBROUTINE COMT2 (PE,TE)                                           COMT2       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90   5
C  COMT2  COMMON TERMS - SUB REGION 2                                    COMT2       3                
C         A 1ST LEVEL SUBROUTINE                                         COMT2       4                
C                                                                        COMT2       5                
C                                                                        COMT2       6                
C     FIRST ARGUMENT IS PRESSURE                                         COMT2       7                
C     SECOND ARGUMENT IS TEMPERATURE                                     COMT2       8                
C                                                                        COMT2       9                
C                                                                        COMT2      10                
      COMMON /CONST2/ BB00, BB01, BB02, BB03, BB04, BB05, BB11, BB12,    COMT2      11                
     1    BB21, BB22, BB23, BB31, BB32, BB41, BB42, BB51, BB52, BB53,    COMT2      12                
     2    BB61, BB62, BB71, BB72, BB81, BB82,  B00,  B61,  B71,  B81,    COMT2      13                
     3     B82, BB90, BB91, BB92, BB93, BB94, BB95, BB96                 COMT2      14                
C                                                                        COMT2      15                
      COMMON /CONSTL/ AL0, AL1, AL2, AL2T2                               COMT2      16                
C                                                                        COMT2      17                
C                                                                        COMT2      18                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  COMT2      19                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  COMT2      20                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  COMT2      21                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          COMT2      22                
C                                                                        COMT2      23                
      COMMON /CURENT / P, T, V, H, S                                     COMT2      24                
C                                                                        COMT2      25                
C                                                                        COMT2      26                
      COMMON /STM2/ THM1  ,THETA ,TH2   ,TH3   ,TH4   ,X1    ,X2    ,    COMT2      27                
     1 X3   ,X4    ,X6    ,X8    ,X10   ,X11   ,X12   ,X13   ,X14   ,    COMT2      28                
     2 X17  ,X18   ,X19   ,X24   ,X25   ,X27   ,X28   ,X32   ,BETA  ,    COMT2      29                
     3 BETA2,BETA3 ,BETA4 ,BETA5 ,BETA6 ,BETA7 ,D4    ,T4    ,D3    ,    COMT2      30                
     4 T3   ,D2    ,T2    ,BETAL ,BOBL  ,BOBLP ,FB    ,BB61F ,BB71F ,    COMT2      31                
     5 BB81F                                                             COMT2      32                
C                                                                        COMT2      33                
      character cccc*6
      data cccc/6H COMT2/
      P = PE                                                             COMT2      34                
      T = TE                                                             COMT2      35                
      IF(P .LT. PMIN .OR. P .GT. PMAX) CALL STER (cccc, 12, P, T)        COMT2      36                
      IF(T .LT. T2MIN.OR. T .GT. TMAX) CALL STER (cccc, 12, P, T)        COMT2      37                
      BETA = P/PCA                                                       COMT2      38                
      THETA = (T + TZA)/TCA                                              COMT2      39                
  150 THM1 = 1.0/THETA                                                   COMT2      40                
      TH2 = THETA * THETA                                                COMT2      41                
      TH3 = THETA * TH2                                                  COMT2      42                
      TH4 = TH2 * TH2                                                    COMT2      43                
      X1  =  DEXP (B00 *(1.0 - THETA))                                   STEAMWV1M0(LIB)-NOV. 1,90   6
      X2  = X1 * X1                                                      COMT2      45                
      X3  = X1 * X2                                                      COMT2      46                
      X4  = X2 * X2                                                      COMT2      47                
      X6  = X3 * X3                                                      COMT2      48                
      X8  = X4 * X4                                                      COMT2      49                
      X10 = X4 * X6                                                      COMT2      50                
      X11 = X1 * X10                                                     COMT2      51                
      X12 = X6 * X6                                                      COMT2      52                
      X13 = X2 * X11                                                     COMT2      53                
      X14 = X6 * X8                                                      COMT2      54                
      X17 = X6 * X11                                                     COMT2      55                
      X18 = X8 * X10                                                     COMT2      56                
      X19 = X8 * X11                                                     COMT2      57                
      X24 = X11 * X13                                                    COMT2      58                
      X25 = X11 * X14                                                    COMT2      59                
      X27 = X13 * X14                                                    COMT2      60                
      X28 = X14 * X14                                                    COMT2      61                
      X32 = X4  * X28                                                    COMT2      62                
      BB61F = BB61 * X1                                                  COMT2      63                
      BB71F = BB71 * X6                                                  COMT2      64                
      BB81F = BB81 * X10                                                 COMT2      65                
      BETA2 = BETA * BETA                                                COMT2      66                
      BETA3 = BETA * BETA2                                               COMT2      67                
      BETA4 = BETA * BETA3                                               COMT2      68                
      BETA5 = BETA * BETA4                                               COMT2      69                
      BETA6 = BETA * BETA5                                               COMT2      70                
      BETA7 = BETA * BETA6                                               COMT2      71                
  200 D4 = 1.0/BETA6 + X27*(B81 * X27 + B82)                             COMT2      72                
      T4 = 27.0 * X27 *(2.0*B81 * X27 + B82)/D4                          COMT2      73                
      D3 = 1.0/BETA5 + B71 * X19                                         COMT2      74                
      T3 = 19.0 * B71 * X19/D3                                           COMT2      75                
      D2 = 1.0/BETA4 + B61 * X14                                         COMT2      76                
      T2 = 14.0 * B61 * X14/D2                                           COMT2      77                
      BETAL = AL0 + THETA*(AL1 + THETA*AL2)                              COMT2      78                
      FB = 10.0 * (AL1 + THETA * AL2T2)/BETAL                            COMT2      79                
  300 BOBL = BETA/BETAL                                                  COMT2      80                
      BOBLP= BOBL**10                                                    COMT2      81                
 6000 RETURN                                                             COMT2      82                
      END                                                                COMT2      83                
