      FUNCTION HVT3(V,T)                                                 HVT3        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  31
C  HVT3  ENTHALPY - SUB REGION 3  H = F(V,T)  A 2ND LEVEL ROUTINE        HVT3        3                
C        ENTRY = H3E(DUMMY)                                              HVT3        4                
C                                                                        HVT3        5                
C                                                                        HVT3        6                
C                                                                        HVT3        7                
C                                                                        HVT3        8                
      COMMON /CONST3/ C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,  HVT3        9                
     1C010,C011,C012, C11, C12, C13, C14, C15, C16, C17, C21, C22, C23,  HVT3       10                
     2 C24, C25, C26, C27, C28, C31, C32, C33, C34, C35, C36, C37, C38,  HVT3       11                
     3 C39,C310, C40, C41, C50, C60, C61, C62, C63, C64, C70, C71, C72,  HVT3       12                
     4 C73, C74, C75, C76, C77, C78, D30, D31, D32, D33, D34, D40, D41,  HVT3       13                
     5 D42, D43, D44, D50, D51, D52                                      HVT3       14                
C                                                                        HVT3       15                
C                                                                        HVT3       16                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  HVT3       17                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  HVT3       18                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  HVT3       19                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          HVT3       20                
C                                                                        HVT3       21                
C                                                                        HVT3       22                
      COMMON /STM3/  THETA ,THL1  ,THLOG ,THM1  ,THM22 ,THM23 ,EX    ,   HVT3       23                
     1 EXM1  ,EXM2  ,EXM5  ,EXM6  ,EXLOG ,X50TH2,X60TH2,OMTH1 ,D3SV  ,   HVT3       24                
     2 VD3SV ,D4SV  ,VD4SV ,D5SV  ,VD5SV ,Y     ,Y2    ,Y3    ,Y30 ,Y31  HVT3       25                
C                                                                        HVT3       26                
C                                                                        HVT3       27                
      DATA  C02T2, C03T3 / -1.554350078D+1, +1.261382256D+1 /            STEAMV1M2 - 2/96             
      DATA  C04T4, C05T5 / -1.107228152D+1, +1.052098535D+1 /            STEAMV1M2 - 2/96             
      DATA  C06T6, C07T7 / -6.878975280D+0, +1.561966595D+0 /            STEAMV1M2 - 2/96             
      DATA  C08T8, C09T9 / +9.300029040D-1, -7.388104896D-1 /            STEAMV1M2 - 2/96             
      DATA C01010, C01111/ +1.941292390D-1, -1.864176336D-2 /            STEAMV1M2 - 2/96             
      DATA  C13T2, C14T3 / -2.407780080D+1, +1.621312266D+1 /            STEAMV1M2 - 2/96             
      DATA  C15T4, C16T5 / -3.975460172D+0, +3.137615910D-1 /            STEAMV1M2 - 2/96             
      DATA  C21T2, C24T2 / -8.597701840D+0, +8.083449180D+0 /            STEAMV1M2 - 2/96             
      DATA  C25T3, C26T4 / +4.666389780D+0, -6.662757400D+0 /            STEAMV1M2 - 2/96             
      DATA  C27T5, C28T2 / +1.624405790D+0, +5.873106500D+1 /            STEAMV1M2 - 2/96             
      DATA  C31T3, C35T2 / +2.384525526D-5, +1.503791908D+1 /            STEAMV1M2 - 2/96             
      DATA  C36T3, C37T4 / -3.784819200D+1, +4.388698480D+0 /            STEAMV1M2 - 2/96             
      DATA  C38T5, C39T6 / +1.060727460D+1, -3.279177396D+0 /            STEAMV1M2 - 2/96             
      DATA C310T2, C40T23/ +1.665750826D+1, +6.347350848D-5 /            STEAMV1M2 - 2/96             
      DATA C40T24, C41T28/ +6.623322624D-5, -1.425407158D-2 /            STEAMV1M2 - 2/96             
      DATA C41T29, C60T3 / -1.476314557D-2, +1.658680600D-1 /            STEAMV1M2 - 2/96             
      DATA  C61T2, C72T2 / -4.672731910D-1, +4.441446416D+1 /            STEAMV1M2 - 2/96             
      DATA  C73T3, C74T4 / -5.406118710D+2, +9.428384880D+3 /            STEAMV1M2 - 2/96             
      DATA  C75T5, C76T6 / -7.311678490D+4, +2.725749978D+5 /            STEAMV1M2 - 2/96             
      DATA  C77T7, C78T8 / -4.937489502D+5, +3.505257142D+5 /            STEAMV1M2 - 2/96             
      DATA    CTA,   CTB / -2.528322967D+0, +5.435179489D+1 /            STEAMV1M2 - 2/96             
      DATA C12M17        / +3.435853127D+0 /                             STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE C02T2,C03T3,C04T4,C05T5,C06T6,C07T7,C08T8,C09T9,C01010,C01111 STEAMV1M2 - 3/96
      SAVE C13T2,C14T3,C15T4,C16T5,C21T2,C24T2,C25T3,C26T4,C27T5,C28T2   STEAMV1M2 - 3/96
      SAVE C31T3,C35T2,C36T3,C37T4,C38T5,C39T6,C310T2,C40T23,C40T24      STEAMV1M2 - 3/96
      SAVE C41T28,C41T29,C60T3,C61T2,C72T2,C73T3,C74T4,C75T5,C76T6       STEAMV1M2 - 3/96
      SAVE C77T7,C78T8,CTA,CTB,C12M17                                    STEAMV1M2 - 3/96
C                                                                        HVT3       50                
      CALL COMT3(V,T)                                                    HVT3       51                
C                                                                        HVT3       52                
      ENTRY H3E(V,T)                                                     STEAMWV1M0(LIB)-NOV. 1,90  32
C                                                                        HVT3       54                
C     THIS ENTRY CAN BE USED ONLY IF COMT3 WAS LAST CALLED WITH THE      HVT3       55                
C     VALUES OF V AND T THAT ARE TO BE ASSUMED HERE.                     HVT3       56                
C                                                                        HVT3       57                
 1000 H0 = CTA - C50 - C11*EX + EXM1*(C02T2 + EXM1*(C03T3 + EXM1*(C04T4+ HVT3       58                
     1 EXM1*(C05T5 + EXM1*(C06T6 + EXM1*(C07T7 + EXM1*(C08T8 +           HVT3       59                
     2 EXM1*(C09T9 + EXM1*(C01010 + EXM1*C01111))))))))) - EXM1*(C12 +   HVT3       60                
     3 EXM1*(C13 + EXM1*(C14 + EXM1*(C15 + EXM1*C16)))) + C12M17*EXLOG   HVT3       61                
      H1 = -C17 - C50 - EX*(C11 + C21T2) + EXM1*(C12 + EXM1*(C13T2 +     HVT3       62                
     1 EXM1*(C14T3 + EXM1*(C15T4 + EXM1*C16T5)))) - 2.0*EXM1*(C22 +      HVT3       63                
     2 EXM1*(C23 + EXM1*(C24 + EXM1*(C25 + EXM1*(C26 + EXM1*C27))))) -   HVT3       64                
     3 C28T2*EXLOG                                                       HVT3       65                
      H2 = -C28 - EX*(C21T2 + C31T3) + EXM2*(C23 + EXM1*(C24T2           HVT3       66                
     1 + EXM1*(C25T3 + EXM1*(C26T4 + EXM1*C27T5)))) - 3.0*EXM1*(C32      HVT3       67                
     2 + EXM1*(C33 + EXM1*(C34 + EXM1*(C35 + EXM1*(C36 + EXM1*(C37       HVT3       68                
     3 + EXM1*(C38 + EXM1*C39))))))) - CTB*EXLOG                         HVT3       69                
      H3 = -C310 - C31T3*EX + EXM1*(-C32 + EXM2*(C34 + EXM1*(C35T2       HVT3       70                
     1 + EXM1*(C36T3 + EXM1*(C37T4 + EXM1*(C38T5 + EXM1*C39T6))))))      HVT3       71                
     2 - C310T2*EXLOG                                                    HVT3       72                
      H4 = (C40T23 + C41T28*EXM5 - (C40T24 + C41T29*EXM5)*THM1)*THM22    HVT3       73                
      H6 = -X60TH2*(C60T3 + THM1*(C61T2 + THM1*(C62+THM1*(-C64*THM1))))  HVT3       74                
      H7 = C70 + THL1*(C71 + C71*THETA + THL1*(C72 + C72T2*THETA         HVT3       75                
     1 + THL1*(C73 + C73T3*THETA + THL1*(C74 + C74T4*THETA               HVT3       76                
     2 + THL1*(C75 + C75T5*THETA + THL1*(C76 + C76T6*THETA               HVT3       77                
     3 + THL1*(C77 + C77T7*THETA + THL1*(C78 + C78T8*THETA))))))))       HVT3       78                
      H8 = 0.0                                                           HVT3       79                
      IF(THETA .GE. 1.0 .OR. EX .GE. 1.0) GO TO 1500                     HVT3       80                
      FTH3 = 3.0/OMTH1                                                   HVT3       81                
      H8A = (D30*(-2.0*Y + FTH3) + EXM1*(D31*(-Y + FTH3) + EXM1*(D32*    HVT3       82                
     1 FTH3 + EXM1*(D33*(Y + FTH3)+EXM1*(D34*(2.0*Y + FTH3)))))) *Y2     HVT3       83                
      FTH4 = 4.0/OMTH1                                                   HVT3       84                
      H8B = (D40*(-3.0*Y + FTH4) + EXM1*(D41*(-2.0*Y + FTH4)             HVT3       85                
     1 + EXM1*(D42*(-Y + FTH4) + EXM1*(D43*FTH4 + EXM1*(D44*(Y           HVT3       86                
     2 + FTH4))))))*Y3                                                   HVT3       87                
      FTH32 = 32.0/OMTH1                                                 HVT3       88                
      H8C = Y31*(D50*(31.0*Y - FTH32) + EX*(D51*(32.0*Y - FTH32)         HVT3       89                
     1 + EX*D52*(33.0*Y - FTH32)))                                       HVT3       90                
      H8 = H8A + H8B - H8C                                               HVT3       91                
 1500 HVT3 = (H0 + THL1*(H1 + THL1*(H2 + THL1*H3)) + H4 + H6 - H7 + H8   HVT3       92                
     1 + ALPHA0)*PV010                                                   HVT3       93                
 2000 RETURN                                                             HVT3       94                
      END                                                                HVT3       95                
