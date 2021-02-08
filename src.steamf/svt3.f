      FUNCTION SVT3(V,T)                                                 SVT3        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  55
C  SVT3   ENTROPY - SUB REGION 3   A 2ND LEVEL SUBROUTINE  S = F(V,T)    SVT3        3                
C         ENTRIES = S3E                                                  SVT3        4                
C         CALLS COMT3,                                                   SVT3        5                
C                                                                        SVT3        6                
C                                                                        SVT3        7                
C                                                                        SVT3        8                
C                                                                        SVT3        9                
      COMMON /CONST3/ C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,  SVT3       10                
     1C010,C011,C012, C11, C12, C13, C14, C15, C16, C17, C21, C22, C23,  SVT3       11                
     2 C24, C25, C26, C27, C28, C31, C32, C33, C34, C35, C36, C37, C38,  SVT3       12                
     3 C39,C310, C40, C41, C50, C60, C61, C62, C63, C64, C70, C71, C72,  SVT3       13                
     4 C73, C74, C75, C76, C77, C78, D30, D31, D32, D33, D34, D40, D41,  SVT3       14                
     5 D42, D43, D44, D50, D51, D52                                      SVT3       15                
C                                                                        SVT3       16                
C                                                                        SVT3       17                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  SVT3       18                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  SVT3       19                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  SVT3       20                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          SVT3       21                
C                                                                        SVT3       22                
C                                                                        SVT3       23                
      COMMON /STM3/  THETA ,THL1  ,THLOG ,THM1  ,THM22 ,THM23 ,EX    ,   SVT3       24                
     1 EXM1  ,EXM2  ,EXM5  ,EXM6  ,EXLOG ,X50TH2,X60TH2,OMTH1 ,D3SV  ,   SVT3       25                
     2 VD3SV ,D4SV  ,VD4SV ,D5SV  ,VD5SV ,Y     ,Y2    ,Y3    ,Y30 ,Y31  SVT3       26                
C                                                                        SVT3       27                
      DATA    C60T2 / +1.105787067D-1 /,  C61T3 / -7.009097865D-1 /,     STEAMV1M2 - 2/96             
     1        C62T4 / +1.478828568D+0 /,  C63T5 / -1.298207735D+0 /,     STEAMV1M2 - 2/96             
     2        C64T6 / +4.096852208D-1 /,  C71T2 / -3.037567430D+2 /,     STEAMV1M2 - 2/96             
     3        C72T3 / +6.662169624D+1 /,  C73T4 / -7.208158280D+2 /,     STEAMV1M2 - 2/96             
     4        C74T5 / +1.178548110D+4 /,  C75T6 / -8.774014188D+4 /,     STEAMV1M2 - 2/96             
     5        C76T7 / +3.180041641D+5 /,  C77T8 / -5.642845147D+5 /,     STEAMV1M2 - 2/96             
     6        C78T9 / +3.943414285D+5 /                                  STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE C60T2,C61T3,C62T4,C63T5,C64T6,C71T2,C72T3,C73T4,C74T5,C75T6   STEAMV1M2 - 3/96
      SAVE C76T7,C77T8,C78T9                                             STEAMV1M2 - 3/96
C                                                                        SVT3       35                
      CALL COMT3(V,T)                                                    SVT3       36                
C                                                                        SVT3       37                
      ENTRY S3E(V,T)                                                     STEAMWV1M0(LIB)-NOV. 1,90  56
C                                                                        SVT3       39                
C     THIS ENTRY TO BE USED ONLY IF COMT3 WAS LAST CALLED WITH THE       SVT3       40                
C     VALUES OF V AND T THAT ARE TO BE ASSUMED HERE.                     SVT3       41                
C                                                                        SVT3       42                
 1000 S1 = C11*EX + EXM1*(C12 + EXM1*(C13 +EXM1*(C14 + EXM1*(C15         SVT3       43                
     1 + EXM1*C16)))) + C17*EXLOG + C50                                  SVT3       44                
      S2 = 2.0*(C21*EX + EXM1*(C22 + EXM1*(C23 + EXM1*(C24               SVT3       45                
     1 + EXM1*(C25 + EXM1*(C26 + C27*EXM1))))) + C28*EXLOG)              SVT3       46                
      S3 = 3.0*(C31*EX + EXM1*(C32 + EXM1*(C33 + EXM1*(C34 + EXM1*(C35   SVT3       47                
     1 + EXM1*(C36 + EXM1*(C37 + EXM1*(C38 + C39*EXM1)))))))+C310*EXLOG) SVT3       48                
      S4 = (C40 + C41*EXM5)*(22.0 - 23.0*THM1)*THM23 - C50* THLOG        SVT3       49                
      S6 = (X60TH2/THETA)*(C60T2 + THM1*(C61T3 + THM1*(C62T4             SVT3       50                
     1 + THM1*(C63T5 + THM1*C64T6))))                                    SVT3       51                
      S7 = C70 + THL1*(C71T2 + THL1*(C72T3 + THL1*(C73T4 + THL1*(C74T5   SVT3       52                
     + + THL1*(C75T6 + THL1*(C76T7 + THL1*(C77T8 + THL1*C78T9)))))))     SVT3       53                
      S8 = 0.0                                                           SVT3       54                
      IF(THETA .GE. 1.0 .OR. EX .GE. 1.0) GO TO 1500                     SVT3       55                
      S8A = Y2*(3.0D0*D3SV + 4.0D0*Y*D4SV)                               SVT3       56                
      S8B = 32.0D0*Y31*D5SV                                              SVT3       57                
      S8 = (S8A + S8B)/OMTH1                                             SVT3       58                
 1500 SVT3 = (-S1 - THL1*(S2 + S3*THL1) + S4 + S6 - S7 + S8-ALPHA1)*PVOT SVT3       59                
 2000 RETURN                                                             SVT3       60                
      END                                                                SVT3       61                
