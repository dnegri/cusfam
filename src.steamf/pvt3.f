      FUNCTION PVT3(V,T)                                                 PVT3        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  40
C  PVT3   PRESSURE - SUBREGION 3   A 2ND LEVEL SUBROUTINE                PVT3        3                
C         WITH ENTRY P3E                                                 PVT3        4                
C         USING COMT3(V,T)                                               PVT3        5                
C                                                                        PVT3        6                
C                                                                        PVT3        7                
C                                                                        PVT3        8                
C                                                                        PVT3        9                
      COMMON /CONST3/ C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,  PVT3       10                
     1C010,C011,C012, C11, C12, C13, C14, C15, C16, C17, C21, C22, C23,  PVT3       11                
     2 C24, C25, C26, C27, C28, C31, C32, C33, C34, C35, C36, C37, C38,  PVT3       12                
     3 C39,C310, C40, C41, C50, C60, C61, C62, C63, C64, C70, C71, C72,  PVT3       13                
     4 C73, C74, C75, C76, C77, C78, D30, D31, D32, D33, D34, D40, D41,  PVT3       14                
     5 D42, D43, D44, D50, D51, D52                                      PVT3       15                
C                                                                        PVT3       16                
C                                                                        PVT3       17                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  PVT3       18                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  PVT3       19                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  PVT3       20                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          PVT3       21                
C                                                                        PVT3       22                
C                                                                        PVT3       23                
      COMMON /STM3/  THETA ,THL1  ,THLOG ,THM1  ,THM22 ,THM23 ,EX    ,   PVT3       24                
     1 EXM1  ,EXM2  ,EXM5  ,EXM6  ,EXLOG ,X50TH2,X60TH2,OMTH1 ,D3SV  ,   PVT3       25                
     2 VD3SV ,D4SV  ,VD4SV ,D5SV  ,VD5SV ,Y     ,Y2    ,Y3    ,Y30 ,Y31  PVT3       26                
C                                                                        PVT3       27                
      DATA    C03T2 / +8.409215040D+0 /,  C04T3 / -8.304211140D+0 /,     STEAMV1M2 - 2/96             
     1        C05T4 / +8.416788280D+0 /,  C06T5 / -5.732479400D+0 /,     STEAMV1M2 - 2/96             
     2        C07T6 / +1.338828510D+0 /,  C08T7 / +8.137525410D-1 /,     STEAMV1M2 - 2/96             
     3        C09T8 / -6.567204352D-1 /, C010T9 / +1.747163151D-1 /,     STEAMV1M2 - 2/96             
     4       C01110 / -1.694705760D-2 /,  C13T2 / -2.407780080D+1 /,     STEAMV1M2 - 2/96             
     5        C14T3 / +1.621312266D+1 /,  C15T4 / -3.975460172D+0 /,     STEAMV1M2 - 2/96             
     6        C16T5 / +3.137615910D-1 /,  C23T2 / -2.832386260D+1 /,     STEAMV1M2 - 2/96             
     7        C24T3 / +1.212517377D+1 /,  C25T4 / +6.221853040D+0 /,     STEAMV1M2 - 2/96             
     8        C26T5 / -8.328446750D+0 /,  C27T6 / +1.949286948D+0 /,     STEAMV1M2 - 2/96             
     9        C33T2 / -1.672306760D+2 /,  C34T3 / +1.075909551D+2 /,     STEAMV1M2 - 2/96             
     A        C35T4 / +3.007583816D+1 /,  C36T5 / -6.308032000D+1 /,     STEAMV1M2 - 2/96             
     B        C37T6 / +6.583047720D+0 /,  C38T7 / +1.485018444D+1 /,     STEAMV1M2 - 2/96             
     C        C39T8 / -4.372236528D+0 /,  C41T5 / -2.545369925D-3 /      STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE C03T2,C04T3,C05T4,C06T5,C07T6,C08T7,C09T8,C010T9,C01110       STEAMV1M2 - 3/96
      SAVE C13T2,C14T3,C15T4,C16T5,C23T2,C24T3,C25T4,C26T5,C27T6         STEAMV1M2 - 3/96
      SAVE C33T2,C34T3,C35T4,C36T5,C37T6,C38T7,C39T8,C41T5               STEAMV1M2 - 3/96
C                                                                        PVT3       41                
      CALL COMT3(V,T)                                                    PVT3       42                
C                                                                        PVT3       43                
      ENTRY P3E(V,T)                                                     STEAMWV1M0(LIB)-NOV. 1,90  41
C                                                                        PVT3       45                
C     THIS ENTRY TO BE USED ONLY IF COMT3 WAS LAST CALLED WITH THE       PVT3       46                
C     VALUES OF V AND T THAT ARE TO BE ASSUMED HERE.                     PVT3       47                
C                                                                        PVT3       48                
 1000 P0 = C01 - EXM2*(C02 + EXM1*(C03T2 + EXM1*(C04T3 + EXM1*(C05T4     PVT3       49                
     1 + EXM1*(C06T5 + EXM1*(C07T6 + EXM1*(C08T7 + EXM1*(C09T8           PVT3       50                
     2 + EXM1*(C010T9 + EXM1*C01110))))))))) + C012*EXM1                 PVT3       51                
      PA = C11 - EXM2*(C12 + EXM1*(C13T2 + EXM1*(C14T3 + EXM1*(C15T4     PVT3       52                
     1 + EXM1*C16T5)))) + C17*EXM1                                       PVT3       53                
      P2 = C21 - EXM2*(C22 + EXM1*(C23T2 + EXM1*(C24T3 + EXM1*(C25T4     PVT3       54                
     1 + EXM1*(C26T5 + EXM1*C27T6))))) + C28*EXM1                        PVT3       55                
      P3 = C31 - EXM2*(C32 + EXM1*(C33T2 + EXM1*(C34T3 + EXM1*(C35T4     PVT3       56                
     1 + EXM1*(C36T5 + EXM1*(C37T6 + EXM1*(C38T7 + EXM1*C39T8)))))))     PVT3       57                
     2 + C310*EXM1                                                       PVT3       58                
      P4 = C41T5*EXM6*THM23                                              PVT3       59                
      P6 = 6.0*X50TH2*(C60 + THM1*(C61 + THM1*(C62 + THM1*(C63           PVT3       60                
     1 + THM1*C64))))                                                    PVT3       61                
      P7 = 0.0                                                           PVT3       62                
      IF(THETA .GE. 1.0 .OR. EX .GE. 1.0) GO TO 1500                     PVT3       63                
      P7A = Y3*(VD3SV + Y*VD4SV)                                         PVT3       64                
      P7B = Y*Y31*VD5SV                                                  PVT3       65                
      P7 = P7A - P7B                                                     PVT3       66                
 1500 PVT3 = (-P0 + THL1*(-PA - THL1*(P2 + P3*THL1) + P4) - P6 + P7)*PCA PVT3       67                
 2000 RETURN                                                             PVT3       68                
      END                                                                PVT3       69                
