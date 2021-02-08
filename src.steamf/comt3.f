      SUBROUTINE COMT3(VE,TE)                                            COMT3       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90   7
C  COMT3  COMMON TERMS - SUB REGION 3                                    COMT3       3                
C         A 1ST LEVEL SUBROUTINE                                         COMT3       4                
C                                                                        COMT3       5                
C                                                                        COMT3       6                
C     FIRST ARGUMENT IS SPECIFIC VOLUME                                  COMT3       7                
C     SECOND ARGUMENT IS TEMPERATURE                                     COMT3       8                
C                                                                        COMT3       9                
C                                                                        COMT3      10                
C                                                                        COMT3      11                
      COMMON /CONST3/ C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,  COMT3      12                
     1C010,C011,C012, C11, C12, C13, C14, C15, C16, C17, C21, C22, C23,  COMT3      13                
     2 C24, C25, C26, C27, C28, C31, C32, C33, C34, C35, C36, C37, C38,  COMT3      14                
     3 C39,C310, C40, C41, C50, C60, C61, C62, C63, C64, C70, C71, C72,  COMT3      15                
     4 C73, C74, C75, C76, C77, C78, D30, D31, D32, D33, D34, D40, D41,  COMT3      16                
     5 D42, D43, D44, D50, D51, D52                                      COMT3      17                
C                                                                        COMT3      18                
C                                                                        COMT3      19                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  COMT3      20                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  COMT3      21                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  COMT3      22                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          COMT3      23                
C                                                                        COMT3      24                
      COMMON /CURENT / P, T, V, H, S                                     COMT3      25                
C                                                                        COMT3      26                
C                                                                        COMT3      27                
      COMMON /STM3/  THETA ,THL1  ,THLOG ,THM1  ,THM22 ,THM23 ,EX    ,   COMT3      28                
     1 EXM1  ,EXM2  ,EXM5  ,EXM6  ,EXLOG ,X50TH2,X60TH2,OMTH1 ,D3SV  ,   COMT3      29                
     2 VD3SV ,D4SV  ,VD4SV ,D5SV  ,VD5SV ,Y     ,Y2    ,Y3    ,Y30 ,Y31  COMT3      30                
C                                                                        COMT3      31                
      DATA D32T2, D33T3 / -5.381798746D+00, +2.721294782D+00/            STEAMV1M2 - 2/96             
      DATA D34T4, D42T2 / -4.555164624D-01, +3.993530724D+00/            STEAMV1M2 - 2/96             
      DATA D43T3, D44T4 / -1.998467104D+00, +3.308344236D-01/            STEAMV1M2 - 2/96             
      DATA D52T2        / +2.310036618D-03 /                             STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE D32T2,D33T3,D34T4,D42T2,D43T3,D44T4,D52T2                     STEAMV1M2 - 3/96
C                                                                        COMT3      36                
      character cccc*6
      data cccc/6H COMT3/
      V = VE                                                             COMT3      37                
      T = TE                                                             COMT3      38                
      IF(V .LT. V3MIN .OR. V .GT. V3MAX) CALL STER(cccc, 42, V, T)       COMT3      39                
      IF(T .LT. T3MIN .OR. T .GT. T3MAX) CALL STER(cccc, 42, V, T)       COMT3      40                
      EX = V/VCA                                                         COMT3      41                
      THETA = (T + TZA)/TCA                                              COMT3      42                
 1000 THL1 = THETA - 1.0                                                 COMT3      43                
      THETA1 = (T1 + TZA)/TCA                                            COMT3      46                
      OMTH1 = 1.0 - THETA1                                               COMT3      47                
      Y = (1.0 - THETA)/OMTH1                                            COMT3      48                
      Y2 = Y*Y                                                           COMT3      49                
      Y3 = Y*Y2                                                          COMT3      50                
      Y30 = Y**30                                                        COMT3      51                
      Y31 = Y * Y30                                                      COMT3      52                
      THLOG = DLOG(THETA)                                                STEAMWV1M0(LIB)-NOV. 1,90  10
      EXM1 = VCA/V                                                       COMT3      54                
      EXM2 = EXM1 * EXM1                                                 COMT3      55                
      EXM4 = EXM2 * EXM2                                                 COMT3      56                
      EXM5 = EXM4 * EXM1                                                 COMT3      57                
      EXM6 = EXM5 * EXM1                                                 COMT3      58                
      EXLOG = DLOG(EX)                                                   STEAMWV1M0(LIB)-NOV. 1,90  11
      D3SV = D30 + EXM1*(D31 + EXM1*(D32 + EXM1*(D33 + EXM1*D34)))       COMT3      60                
      VD3SV = EXM2*(D31 + EXM1*(D32T2 + EXM1*(D33T3 + EXM1*D34T4)))      COMT3      61                
      D4SV = D40 + EXM1*(D41 + EXM1*(D42 + EXM1*(D43 + EXM1*D44)))       COMT3      62                
      VD4SV = EXM2*(D41 + EXM1*(D42T2 + EXM1*(D43T3 + EXM1*D44T4)))      COMT3      63                
      D5SV = D50 + EX*(D51 + EX*D52)                                     COMT3      64                
      VD5SV = D51 + D52T2*EX                                             COMT3      65                
      THM1 = 1.0/THETA                                                   COMT3      66                
      THM22 = THM1**22                                                   COMT3      67                
      THM23 = THM22*THM1                                                 COMT3      68                
      X50TH2 = EX**5/(THETA*THETA)                                       COMT3      69                
      X60TH2 = EX*X50TH2                                                 COMT3      70                
 6000 RETURN                                                             COMT3      71                
      END                                                                COMT3      72                
