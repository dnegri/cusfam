      FUNCTION CPVT3(V,T)                                                CPVT3       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  21
C                                                                        CPVT3       3                
CCPVT3      SPECIFIC HEAT - SUB REGION 3                                 CPVT3       4                
C           A 2ND LEVEL SUBROUTINE                                       CPVT3       5                
C           WITH ENTRY CP3E                                              CPVT3       6                
C                                                                        CPVT3       7                
C     FIRST ARGUMENT IS SPECIFIC VOLUME                                  CPVT3       8                
C     SECOND ARGUMENT IS TEMPERATURE                                     CPVT3       9                
C     RETURNS WITH SPECIFIC HEAT                                         CPVT3      10                
C                                                                        CPVT3      11                
      COMMON /CONST3/ C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,  CPVT3      12                
     1C010,C011,C012, C11, C12, C13, C14, C15, C16, C17, C21, C22, C23,  CPVT3      13                
     2 C24, C25, C26, C27, C28, C31, C32, C33, C34, C35, C36, C37, C38,  CPVT3      14                
     3 C39,C310, C40, C41, C50, C60, C61, C62, C63, C64, C70, C71, C72,  CPVT3      15                
     4 C73, C74, C75, C76, C77, C78, D30, D31, D32, D33, D34, D40, D41,  CPVT3      16                
     5 D42, D43, D44, D50, D51, D52                                      CPVT3      17                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  CPVT3      18                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  CPVT3      19                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  CPVT3      20                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          CPVT3      21                
      COMMON /STM3/  THETA ,THL1  ,THLOG ,THM1  ,THM22 ,THM23 ,EX    ,   CPVT3      22                
     1 EXM1  ,EXM2  ,EXM5  ,EXM6  ,EXLOG ,X50TH2,X60TH2,OMTH1 ,D3SV  ,   CPVT3      23                
     2 VD3SV ,D4SV  ,VD4SV ,D5SV  ,VD5SV ,Y     ,Y2    ,Y3    ,Y30 ,Y31  CPVT3      24                
      DATA C02T2,  C03T6 /-1.554350078D1, +2.522764512D1 /               STEAMV1M2 - 2/96
      DATA C04T12, C05T20 /-3.321684456D1, +4.208394140D1 /              STEAMV1M2 - 2/96
      DATA C06T30, C07T42 /-3.439487640D1, +9.371799570D0 /              STEAMV1M2 - 2/96
      DATA C08T56, C09T72 /+6.510020328D0, -5.910483917D0 /              STEAMV1M2 - 2/96
      DATA C01090, C011T  /+1.747163151D0, -1.864176336D-1 /             STEAMV1M2 - 2/96
      DATA C12T2 /+2.473589100D1 /                                       STEAMV1M2 - 2/96
      DATA C13T2, C14T3     /-2.407780080D1, +1.621312266D1 /            STEAMV1M2 - 2/96
      DATA C13T6, C14T12 /-7.223340240D1, 6.4852490064D1 /               STEAMV1M2 - 2/96
      DATA C15T4, C16T5 /-3.975460172D0, +3.137615910D-1 /               STEAMV1M2 - 2/96
      DATA C15T20, C16T30 /-1.987730086D1, 1.882569546D0 /               STEAMV1M2 - 2/96
      DATA C22T2, C23T6 /+8.628610760D1, -8.497158780D1 /                STEAMV1M2 - 2/96
      DATA C23T2, C24T3 /-2.83238626D1, +1.212517377D1 /                 STEAMV1M2 - 2/96
      DATA C24T12,C25T20 /+4.850069508D1, +3.110926520D1 /               STEAMV1M2 - 2/96
      DATA C25T4, C26T5 /+6.221853040D0, -8.328446750D0 /                STEAMV1M2 - 2/96
      DATA C26T30, C27T42 /-4.997068050D1, +1.364500864D1 /              STEAMV1M2 - 2/96
      DATA C27T6 /+1.949286948D0 /                                       STEAMV1M2 - 2/96
      DATA C32T2, C33T6 /+1.617719494D2, -5.016920280D2 /                STEAMV1M2 - 2/96
      DATA C33T2, C34T3 /-1.672306760D2, +1.075909551D2 /                STEAMV1M2 - 2/96
      DATA C34T12, C35T20 /+4.303638204D2, +1.503791908D2 /              STEAMV1M2 - 2/96
      DATA C35T4, C36T5 /+3.007583816D1, -6.308032000D1 /                STEAMV1M2 - 2/96
      DATA C36T30, C37T42 /-3.784819200D2, +4.608133404D1 /              STEAMV1M2 - 2/96
      DATA C37T6, C38T7 /+6.583047720D0, +1.485018444D1 /                STEAMV1M2 - 2/96
      DATA C38T56, C39T72 /+1.188014755D2, -3.935012875D1 /              STEAMV1M2 - 2/96
      DATA C39T8 /-4.372236528D0 /                                       STEAMV1M2 - 2/96
      DATA C41T5, C41T30 /-2.545369925D-3, -1.527221955D-2 /             STEAMV1M2 - 2/96
      DATA C60T2, C60T6  /+1.105787067D-1, +3.317361201D-1 /             STEAMV1M2 - 2/96
      DATA C61T3, C61T12 /-7.009097865D-1, -2.803639146D0 /              STEAMV1M2 - 2/96
      DATA C62T4, C62T20 /+1.478828568D0, +7.394142840D0 /               STEAMV1M2 - 2/96
      DATA C63T5, C63T30 /-1.298207737D0, -7.789246410D0 /               STEAMV1M2 - 2/96
      DATA C64T6, C64T42 /+4.096852208D-1, +2.867796546D0 /              STEAMV1M2 - 2/96
      DATA C71T2, C72T6  /-3.037567430D+2, +1.332433925D2 /              STEAMV1M2 - 2/96
      DATA C73T12, C74T20 /-2.162447484D3, +4.714192440D4 /              STEAMV1M2 - 2/96
      DATA C75T30, C76T42 /-4.387007094D5, +1.908024985D6 /              STEAMV1M2 - 2/96
      DATA C77T56, C78T72 /-3.949991602D6, +3.154731428D6 /              STEAMV1M2 - 2/96
      DATA D31T2, D32T6 /+7.052779750D0, -1.614539624D1 /                STEAMV1M2 - 2/96
      DATA D33T12, D34T20 /+1.088517913D1, -2.277582312D0 /              STEAMV1M2 - 2/96
      DATA D41T2, D42T6 /-5.285555486D0, +1.198059217D1 /                STEAMV1M2 - 2/96
      DATA D43T12, D44T20 /-7.993868416D0, +1.654172118D0 /              STEAMV1M2 - 2/96
      DATA D52T2 /+2.310036618D-3 /                                      STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE C02T2,C03T6,C04T12,C05T20,C06T30,C07T42,C08T56,C09T72         STEAMV1M2 - 3/96
      SAVE C01090,C011T,C12T2,C13T2,C14T3,C13T6,C14T12,C15T4,C16T5       STEAMV1M2 - 3/96
      SAVE C15T20,C16T30,C22T2,C23T6,C23T2,C24T3,C24T12,C25T20           STEAMV1M2 - 3/96
      SAVE C25T4,C26T5,C26T30,C27T42,C27T6,C32T2,C33T6,C33T2,C34T3       STEAMV1M2 - 3/96
      SAVE C34T12,C35T20,C35T4,C36T5,C36T30,C37T42,C37T6,C38T7,C38T56    STEAMV1M2 - 3/96
      SAVE C39T72,C39T8,C41T5,C41T30,C60T2,C60T6,C61T3,C61T12            STEAMV1M2 - 3/96
      SAVE C62T4,C62T20,C63T5,C63T30,C64T6,C64T42,C71T2,C72T6,C73T12     STEAMV1M2 - 3/96
      SAVE C74T20,C75T30,C76T42,C77T56,C78T72,D31T2,D32T6,D33T12,D34T20  STEAMV1M2 - 3/96
      SAVE D41T2,D42T6,D43T12,D44T20,D52T2                               STEAMV1M2 - 3/96
C                                                                        CPVT3      64                
      CALL COMT3(V,T)                                                    CPVT3      65                
C                                                                        CPVT3      66                
      ENTRY CP3E(V,T)                                                    STEAMWV1M0(LIB)-NOV. 1,90  22
C                                                                        CPVT3      68                
C     THIS ENTRY CAN BE USED ONLY IF COMT3 WAS LAST CALLED WITH THE      CPVT3      69                
C     VALUES OF V AND T THAT ARE TO BE ASSUMED HERE.                     CPVT3      70                
C                                                                        CPVT3      71                
      EXM3 = EXM1*EXM2                                                   CPVT3      72                
 1000  CP2 = 2.0D0*(C21*EX + EXM1*(C22 + EXM1*(C23 + EXM1*(C24           STEAMV1M2 - 2/96
     1 + EXM1*(C25 + EXM1*(C26 + EXM1*C27))))) + C28*EXLOG)              CPVT3      74                
      CP3 = 6.0D0*(C31*EX + EXM1*(C32 + EXM1*(C33 + EXM1*(C34            STEAMV1M2 - 2/96
     1 + EXM1*(C35 + EXM1*(C36 + EXM1*(C37 + EXM1*(C38                   CPVT3      76                
     2 + EXM1*C39))))))) + C310*EXLOG)*THL1                              CPVT3      77                
      CP4 = (C40 + C41*EXM5)*(506.0D0 - 552.0D0*THM1)*THM23 + C50        STEAMV1M2 - 2/96
      CP6 = (X60TH2/(THETA*THETA))*(C60T6 + THM1*(C61T12 + THM1*(C62T20  CPVT3      79                
     1 + THM1*(C63T30 + THM1*C64T42))))                                  CPVT3      80                
      CP7 = C71T2 + THL1*(C72T6 + THL1*(C73T12 + THL1*(C74T20 + THL1*    CPVT3      81                
     1 (C75T30 + THL1*(C76T42 + THL1*(C77T56 + THL1*C78T72))))))         CPVT3      82                
      CP = -THETA*(CP2 + CP3 + CP6 + CP7) - CP4                          CPVT3      83                
      CPN1 = C11 - EXM2*(C12 + EXM1*(C13T2 + EXM1*(C14T3 + EXM1*(C15T4   CPVT3      84                
     1 + EXM1*C16T5)))) + C17*EXM1                                       CPVT3      85                
      CPN2 = 2.0D0*(C21 - EXM2*(C22 + EXM1*(C23T2 + EXM1*(C24T3 + EXM1*( STEAMV1M2 - 2/96
     1 C25T4 + EXM1*(C26T5 + EXM1*C27T6))))) + C28*EXM1)*THL1            CPVT3      87                
      CPN3 = 3.0D0*(C31 - EXM2*(C32 + EXM1*(C33T2 + EXM1*(C34T3 + EXM1*( STEAMV1M2 - 2/96
     1 C35T4 + EXM1*(C36T5 + EXM1*(C37T6 + EXM1*(C38T7 + EXM1*C39T8))))) CPVT3      89                
     2)) + C310*EXM1)*THL1*THL1                                          CPVT3      90                
      CPN4 = C41T5*EXM6*(-22.0D0 + 23.0D0*THM1)*THM23                    STEAMV1M2 - 2/96
      CPN6 = 6.0D0*X50TH2*THM1*(C60T2 + THM1*(C61T3 + THM1*(C62T4        STEAMV1M2 - 2/96
     1 + THM1*(C63T5  + THM1*C64T6))))                                   CPVT3      93                
      CPN = CPN1 + CPN2 + CPN3 - CPN4 - CPN6                             CPVT3      94                
      CPD0 = EXM3*(C02T2 + EXM1*(C03T6 + EXM1*(C04T12 + EXM1*(C05T20     CPVT3      95                
     1 + EXM1*(C06T30 + EXM1*(C07T42 + EXM1*(C08T56 + EXM1*(C09T72       CPVT3      96                
     2 + EXM1*(C01090 + EXM1*C011T))))))))) - C012*EXM2                  CPVT3      97                
      CPD1 = (EXM3*(C12T2 + EXM1*(C13T6 + EXM1*(C14T12 + EXM1*(C15T20    CPVT3      98                
     1 + EXM1*C16T30)))) - C17*EXM2)                                     CPVT3      99                
      CPD2 = (EXM3*(C22T2 + EXM1*(C23T6 + EXM1*(C24T12 + EXM1*(C25T20    CPVT3     100                
     1 + EXM1*(C26T30 + EXM1*C27T42))))) - C28*EXM2)                     CPVT3     101                
      CPD3 = EXM3*(C32T2 + EXM1*(C33T6 + EXM1*(C34T12 + EXM1 *(C35T20    CPVT3     102                
     1 + EXM1*(C36T30 + EXM1*(C37T42 + EXM1*(C38T56 + EXM1*C39T72))))))) CPVT3     103                
     2 - C310*EXM2                                                       CPVT3     104                
      CPD4 = C41T30*EXM5*EXM2*(1.0D0 - THM1)*THM22                       STEAMV1M2 - 2/96
      CPD6 = 30.0D0*X50TH2*EXM1*(C60 + THM1*(C61 + THM1*(C62             STEAMV1M2 - 2/96
     1 + THM1*(C63 + THM1*C64))))                                        CPVT3     107                
      CPD = CPD0 + THL1*(CPD1 + THL1*(CPD2 + THL1*CPD3)) + CPD4 + CPD6   CPVT3     108                
      IF (THETA .GE. 1.0D0 .OR.EX .GE. 1.0D0) GO TO 1500                 STEAMV1M2 - 2/96
      PHID = (1.0D0/(OMTH1*OMTH1))*(6.0D0*Y*D3SV + 12.0D0*Y2*D4SV        STEAMV1M2 - 2/96
     1 + 992.0D0*Y30*D5SV)                                               STEAMV1M2 - 2/96
      PHIDN = (1.0D0/OMTH1)*(3.0D0*Y2*VD3SV + 4.0D0*Y3*VD4SV             STEAMV1M2 - 2/96
     1 - 32.0D0*Y31*VD5SV)                                               CPVT3     113                
      VVD3V = Y3*EXM3*(D31T2 + EXM1*(D32T6 + EXM1*(D33T12                CPVT3     114                
     1 + EXM1*D34T20)))                                                  CPVT3     115                
      VVD4V = Y*Y3*EXM3*(D41T2 + EXM1*(D42T6 + EXM1*(D43T12              CPVT3     116                
     1 + EXM1*D44T20)))                                                  CPVT3     117                
      VVD5V = Y*Y31*D52T2                                                CPVT3     118                
      PHIDD = VVD3V + VVD4V + VVD5V                                      CPVT3     119                
      CP = CP - THETA*PHID                                               CPVT3     120                
      CPN = CPN + PHIDN                                                  CPVT3     121                
      CPD = CPD + PHIDD                                                  CPVT3     122                
 1500  CPVT3 = (CP + THETA*CPN*CPN/CPD)*PVOT                             CPVT3     123                
 2000  RETURN                                                            CPVT3     124                
      END                                                                CPVT3     125                
