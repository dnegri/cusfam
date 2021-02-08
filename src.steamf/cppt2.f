      FUNCTION CPPT2(P,T)                                                CPPT2       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  19
CCPPT2     SPECIFIC HEAT - SUB REGION 2                                  CPPT2       3                
C          A 2ND LEVEL SUBROUTINE                                        CPPT2       4                
C          WITH ENTRY CP2E                                               CPPT2       5                
C                                                                        CPPT2       6                
C     FIRST ARGRMENT IS PRESSURE                                         CPPT2       7                
C     SECOND ARGUMENT IS TEMPERATURE                                     CPPT2       8                
C     RETURNS WITH SPECIFIC HEAT                                         CPPT2       9                
C                                                                        CPPT2      10                
      COMMON /CONST2/ BB00, BB01, BB02, BB03, BB04, BB05, BB11, BB12,    CPPT2      11                
     1    BB21, BB22, BB23, BB31, BB32, BB41, BB42, BB51, BB52, BB53,    CPPT2      12                
     2    BB61, BB62, BB71, BB72, BB81, BB82,  B00,  B61,  B71,  B81,    CPPT2      13                
     3     B82, BB90, BB91, BB92, BB93, BB94, BB95, BB96                 CPPT2      14                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  CPPT2      15                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  CPPT2      16                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  CPPT2      17                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          CPPT2      18                
      COMMON/CONSTL/AL0,AL1,AL2,AL2T2                                    CPPT2      19                
      COMMON /STM2/ THM1  ,THETA ,TH2   ,TH3   ,TH4   ,X1    ,X2    ,    CPPT2      20                
     1 X3   ,X4    ,X6    ,X8    ,X10   ,X11   ,X12   ,X13   ,X14   ,    CPPT2      21                
     2 X17  ,X18   ,X19   ,X24   ,X25   ,X27   ,X28   ,X32   ,BETA  ,    CPPT2      22                
     3 BETA2,BETA3 ,BETA4 ,BETA5 ,BETA6 ,BETA7 ,D4    ,T4    ,D3    ,    CPPT2      23                
     4 T3   ,D2    ,T2    ,BETAL ,BOBL  ,BOBLP ,FB    ,BB61F ,BB71F ,    CPPT2      24                
     5 BB81F                                                             CPPT2      25                
C                                                                        CPPT2      26                
      DATA BB03T2,BB04T6 /+8.661325668D-1, -3.928627018D0 /              STEAMV1M2 - 2/96             
      DATA BB0512 /+1.027821847D0 /                                      STEAMV1M2 - 2/96             
      DATA BB11A, BB12T9 /+1.127293530D1, 1.250085421D1 /                STEAMV1M2 - 2/96             
      DATA BB21A, BB22T4 / 2.718393802D+1, 1.045868357D-1 /              STEAMV1M2 - 2/96             
      DATA BB31A, BB32A /+1.464777725D2, 1.069036614D1 /                 STEAMV1M2 - 2/96             
      DATA BB41A, BB42A /-3.734585442D2, -1.734117018D1 /                STEAMV1M2 - 2/96
      DATA BB51A, BB52A /+6.101044848D2, -4.044893844D2 /                STEAMV1M2 - 2/96
      DATA BB53A /+1.195212166D2 /                                       STEAMV1M2 - 2/96
      DATA BB6211, BB62P /-1.085389155D+0, -1.193928070D1 /              STEAMV1M2 - 2/96
      DATA BB7218, BB72P / -1.045698840D+0, -1.882257912D1 /             STEAMV1M2 - 2/96
      DATA BB8214, BB82P / +7.994306109D-3, +1.119202855D-1 /            STEAMV1M2 - 2/96
      DATA BB92T2, BB92T4 /+8.253214438D3, 1.650642876D4 /               STEAMV1M2 - 2/96
      DATA BB93T3, BB93T9 /-1.952463503D4,-5.857390509D4 /               STEAMV1M2 - 2/96
      DATA BB94T4, BB9416 /+2.298393622D4, 9.193574486D4 /               STEAMV1M2 - 2/96
      DATA BB95T5, BB9525 /-1.346544183D4,-6.732720913D4 /               STEAMV1M2 - 2/96
      DATA BB96T6, BB9636 /3.141431174D3, 1.884858704D4/                 STEAMV1M2 - 2/96
      DATA B00T14, B00T19 / 1.068666667D1, 1.450333333D1/                STEAMV1M2 - 2/96
      DATA B00SQ, B00T27 / 5.826777777D-1, 2.061000000D1 /               STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE BB03T2,BB04T6,BB0512,BB11A,BB12T9,BB21A,BB22T4,BB31A,BB32A    STEAMV1M2 - 3/96
      SAVE BB41A,BB42A,BB51A,BB52A,BB53A,BB6211,BB62P,BB7218,BB72P       STEAMV1M2 - 3/96
      SAVE BB8214,BB82P,BB92T2,BB92T4,BB93T3,BB93T9,BB94T4,BB9416        STEAMV1M2 - 3/96
      SAVE BB95T5,BB9525,BB96T6,BB9636,B00T14,B00T19,B00SQ,B00T27        STEAMV1M2 - 3/96
C                                                                        CPPT2      45                
      CALL COMT2 (P,T)                                                   CPPT2      46                
C                                                                        CPPT2      47                
      ENTRY CP2E(P,T)                                                    STEAMWV1M0(LIB)-NOV. 1,90  20
C                                                                        CPPT2      49                
C     THIS ENTRY CAN BE USED ONLY IF COMT2 WAS LAST CALLED WITH THE      CPPT2      50                
C     VALUES OF P AND T THAT ARE TO BE ASSUMED HERE.                     CPPT2      51                
C                                                                        CPPT2      52                
 1000  CP0 = BB00/THETA - (BB03T2 + THETA*(BB04T6+THETA*BB0512))         CPPT2      53                
      CP1 = X3*(BB11A*X10 + BB12T9)                                      CPPT2      54                
      CP2 = X1*(BB21A*X17 + BB22T4*X1 + BB23)                            CPPT2      55                
      CP3 = X10*(BB31A*X8 + BB32A)                                       CPPT2      56                
      CP4 = X14*(BB41A*X11 + BB42A)                                      CPPT2      57                
      CP5 = X24*(BB51A*X8 + BB52A*X4 + BB53A)                            CPPT2      58                
      CPPA = CP0 + BETA*(CP1 + BETA*(CP2 + BETA*(CP3 + BETA*(CP4         CPPT2      59                
     1 + BETA*CP5))))*B00SQ                                              CPPT2      60                
      BB61F = BB61*X1                                                    CPPT2      61                
      BB71F = BB71*X6                                                    CPPT2      62                
      BB81F = BB81*X10                                                   CPPT2      63                
      B4X11 = BETA4*X11                                                  CPPT2      64                
      B5X18 = BETA5*X18                                                  CPPT2      65                
      B6X14 = BETA6*X14                                                  CPPT2      66                
      F60 = B4X11*(BB61F + BB62)                                         CPPT2      67                
      F70 = B5X18*(BB71F + BB72)                                         CPPT2      68                
      F80 = B6X14*(BB81F + BB82)                                         CPPT2      69                
      F60P = -B00*B4X11*(12.0D0*BB61F + BB6211)                          STEAMV1M2 - 2/96
      F70P = -B00*B5X18*(24.0D0*BB71F + BB7218)                          STEAMV1M2 - 2/96
      F80P = -B00*B6X14*(24.0D0*BB81F + BB8214)                          STEAMV1M2 - 2/96
      F60PP =   B00SQ*B4X11*(144.0D0*BB61F + BB62P)                      STEAMV1M2 - 2/96
      F70PP =   B00SQ*B5X18*(576.0D0*BB71F + BB72P)                      STEAMV1M2 - 2/96
      F80PP =   B00SQ*B6X14*(576.0D0*BB81F + BB82P)                      STEAMV1M2 - 2/96
      FF61 =  B61*X14*BETA4                                              CPPT2      76                
      FF71 = B71*X19*BETA5                                               CPPT2      77                
      FF81 =  B81*X27*BETA6*X27                                          CPPT2      78                
      FF82 =  B82*X27*BETA6                                              CPPT2      79                
      G60 = 1.0D0 + FF61                                                 STEAMV1M2 - 2/96
      G70 = 1.0D0 + FF71                                                 STEAMV1M2 - 2/96
      G80 = 1.0D0 + FF81 + FF82                                          STEAMV1M2 - 2/96
      G60P = -B00T14*FF61                                                CPPT2      83                
      G70P = -B00T19*FF71                                                CPPT2      84                
      G80P = -B00T27*(2.0D0*FF81 + FF82)                                 STEAMV1M2 - 2/96
      G60PP = -B00T14*G60P                                               CPPT2      86                
      G70PP = -B00T19*G70P                                               CPPT2      87                
      G80PP = 729.0D0*B00SQ*(4.0D0*FF81 + FF82)                          STEAMV1M2 - 2/96
      CP60 = (F60PP + ((-2.0D0*F60P*G60P - F60*G60PP)                    STEAMV1M2 - 2/96
     1 + 2.0D0*F60*G60P*G60P/G60)/G60)/G60                               STEAMV1M2 - 2/96
      CP70 = (F70PP + ((-2.0D0*F70P*G70P - F70*G70PP)                    STEAMV1M2 - 2/96
     1 + 2.0D0*F70*G70P*G70P/G70)/G70)/G70                               STEAMV1M2 - 2/96
      CP80 = (F80PP + ((-2.0D0*F80P*G80P - F80*G80PP)                    STEAMV1M2 - 2/96
     1 + 2.0D0*F80*G80P*G80P/G80)/G80)/G80                               STEAMV1M2 - 2/96
      CPPB = CP60 + CP70 + CP80                                          CPPT2      95                
      BP = AL1 + AL2T2*THETA                                             CPPT2      96                
      BOBL11 = BOBL*BOBLP                                                CPPT2      97                
      SUMX = BB90 + X1*(BB91 + X1*(BB92 + X1*(BB93 + X1*(BB94 + X1*(BB95 CPPT2      98                
     1 + X1*BB96)))))                                                    CPPT2      99                
      SUMVX = B00*X1*(BB91 + X1*(BB92T2 + X1*(BB93T3                     CPPT2     100                
     1 + X1*(BB94T4 + X1*(BB95T5 + BB96T6*X1)))))                        CPPT2     101                
      VVSUMX =   B00SQ*X1*(BB91+ X1*(BB92T4 + X1*(BB93T9                 CPPT2     102                
     1 + X1*(BB9416 + X1*(BB9525 + X1*BB9636)))))                        CPPT2     103                
      CPP = BOBL11*(((110.0D0/BETAL)*BP*BP - 10.0D0*AL2T2)*SUMX          STEAMV1M2 - 2/96
     1 + 20.0D0*BP*SUMVX + BETAL*VVSUMX)                                 STEAMV1M2 - 2/96
      CPPT2 = THETA*(CPPA + CPPB - CPP)*PVOT                             CPPT2     106                
 2000  RETURN                                                            CPPT2     107                
      END                                                                CPPT2     108                
