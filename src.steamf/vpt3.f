      FUNCTION VPT3(PRES,TEMP)                                           VPT3        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  73
C  VPT3   SPECIFIC VOLUME - SUBREGION 3  A 3RD LEVEL ROUTINE             VPT3        3                
C         WITH ENTRIES VPT3L, VPT3D, VPTF3, AND VPTG3                    VPT3        4                
C         CALLS PSATT, STER, PVT3                                        VPT3        5                
C                                                                        VPT3        6                
C                                                                        VPT3        7                
C                                                                        VPT3        8                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  VPT3        9                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  VPT3       10                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  VPT3       11                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          VPT3       12                
C                                                                        VPT3       13                
      character*6 PT3,PTL3,PTD3,F3,G3,CODE
      DATA PT3 /6H VPT3 /, PTL3 /6H VPT3L /, PTD3 /6H VPT3D /            VPT3       14                
      DATA  F3 /6H VPTF3/,   G3 /6H VPTG3 /                              VPT3       15
C                                                                        STEAMV1M2 - 3/96
      SAVE PT3,PTL3,PTD3,F3,G3                                           STEAMV1M2 - 3/96
C                                                                        VPT3       16                
      CODE = PT3                                                         VPT3       17                
      GO TO 100                                                          VPT3       18                
C                                                                        VPT3       19                
C                                                                        VPT3       20                
      ENTRY VPT3L(PRES,TEMP)                                             STEAMWV1M0(LIB)-NOV. 1,90  74
C     A 3RD LEVEL ENTRY                                                  VPT3       22                
C                                                                        VPT3       23                
      CODE = PTL3                                                        VPT3       24                
      GO TO 100                                                          VPT3       25                
C                                                                        VPT3       26                
      ENTRY VPT3D(PRES,TEMP)                                             STEAMWV1M0(LIB)-NOV. 1,90  75
C     A 3RD LEVEL ENTRY                                                  VPT3       28                
C                                                                        VPT3       29                
      CODE = PTD3                                                        VPT3       30                
  100 P = PRES                                                           VPT3       31                
      T = TEMP                                                           VPT3       32                
      VUP = V3MAX                                                        VPT3       33                
      VLO = V3MIN                                                        VPT3       34                
      IF(CODE .EQ. PTL3) GO TO 400                                       VPT3       35                
      IF(CODE .EQ. PTD3) GO TO 500                                       VPT3       36                
      PLINE = PCA                                                        VPT3       37                
      IF( T .LT. TC) PLINE = PSL(T)                                      VPT3       38                
      IF(P .GE. PLINE) GO TO 400                                         VPT3       39                
      GO TO 500                                                          VPT3       40                
C                                                                        VPT3       41                
      ENTRY VPTF3(PRES,TEMP)                                             STEAMWV1M0(LIB)-NOV. 1,90  76
C     A 3RD LEVEL ENTRY                                                  VPT3       43                
C                                                                        VPT3       44                
      P = PRES                                                           VPT3       45                
      T = TEMP                                                           VPT3       46                
      CODE = F3                                                          VPT3       47                
      B = T/TC                                                           VPT3       48                
      VUP = VCA*B*B                                                      VPT3       49                
      VLO = V3MIN                                                        VPT3       50                
      PLINE = PSL(T)                                                     VPT3       51                
      DP = 1.0D-6*PLINE                                                  VPT3       52                
      IF(DP .LT. 5.0D-7) DP = 5.0D-7                                     VPT3       53                
      IF( P .LT. (PLINE - DP)) CALL STER(CODE, 12, P, T)                 VPT3       54                
  400 V = (0.077D0 + 92.8D0/P)*(T - 482.0)/(T + 58.0D0)                  VPT3       55                
      GO TO 600                                                          VPT3       56                
C                                                                        VPT3       57                
      ENTRY VPTG3(PRES,TEMP)                                             STEAMWV1M0(LIB)-NOV. 1,90  77
C     A 3RD LEVEL ENTRY                                                  VPT3       59                
C                                                                        VPT3       60                
      P = PRES                                                           VPT3       61                
      T = TEMP                                                           VPT3       62                
      CODE = G3                                                          VPT3       63                
      B = TC/T                                                           VPT3       64                
      VUP = V3MAX                                                        VPT3       65                
      VLO = VCA*B*B                                                      VPT3       66                
      PLINE = PSL(T)                                                     VPT3       67                
      DP = 1.0D-6*PLINE                                                  VPT3       68                
      IF(DP .LT. 5.0D-7) DP = 5.0D-7                                     VPT3       69                
      IF( P .GT. (PLINE + DP)) CALL STER(CODE, 12, P, T)                 VPT3       70                
  500 V = 0.3D0*(T + TZA)/P                                              VPT3       71                
  600 IF(P .LT. P3MIN .OR. P .GT. PMAX) CALL STER(CODE, 12, P, T)        VPT3       72                
      DF = 0.25D0                                                        VPT3       73                
      PCTP = 0.0D0                                                       VPT3       74                
      DO 4000 I = 1,100                                                  VPT3       75                
      IF(V .LE. VUP) GO TO 800                                           VPT3       76                
      V = VUP                                                            VPT3       77                
      DF = 1.0D0                                                         VPT3       78                
      GO TO 900                                                          VPT3       79                
  800 IF(V .GE. VLO) GO TO 900                                           VPT3       80                
      V = VLO                                                            VPT3       81                
      DF = 1.0D0                                                         VPT3       82                
  900 PCALC = PVT3(V,T)                                                  VPT3       83                
      PCT = (PCALC - P)/P                                                VPT3       84                
  910 IF( DABS(PCT) .LE. 1.0D-7) GO TO 6000                              STEAMV1M2 - 2/96             
      IF(I .LT. 2) GO TO 970                                             VPT3       86                
      IF(PCTP*PCT) 940,940,920                                           VPT3       87                
  920 IF( DABS(PCT) .GT. DABS(0.3D0*PCTP)) GO TO 960                     STEAMV1M2 - 2/96
      GO TO 950                                                          VPT3       89                
  940 DF = 0.67D0*DF                                                     VPT3       90                
  950 DV = (V - VPREV)*(P - PCALC)/(PCALC - PPREV)                       VPT3       91                
      GO TO 2000                                                         VPT3       92                
  960 DF = 1.5D0*DF                                                      VPT3       93                
  970 DV = V*PCT*DF                                                      VPT3       94                
 2000 VPREV = V                                                          VPT3       95                
      PPREV = PCALC                                                      VPT3       96                
      PCTP = PCT                                                         VPT3       97                
      V = V + DV                                                         VPT3       98                
 4000 CONTINUE                                                           VPT3       99                
      CALL STER( CODE, -12, P, T)                                        VPT3      100                
 6000 VPT3 = V                                                           VPT3      101                
      RETURN                                                             VPT3      102                
      END                                                                VPT3      103                
