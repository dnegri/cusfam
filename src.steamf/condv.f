      FUNCTION CONDV(P,TIN)                                              CONDV       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  12
CCOND THERMAL CONDUCTIVITY AS A FUNCTION OF PRESSURE AND TEMP.           CONDV       3                
C                                                                        CONDV       4                
C                                                                        CONDV       5                
C     FIRST ARGUMENT IS PRESSURE                                         CONDV       6                
C     SECOND ARGUMENT IS TEMPERATURE                                     CONDV       7                
C     RETURNS WITH THERMAL CONDUCTIVITY IN BTU/HR-FT-DEGF                CONDV       8                
C     CONDV IS ENTRY FOR VAPOR REGION                                    CONDV       9                
C     CONDL IS ENTRY FOR LIQUID REGION                                   CONDV      10                
C                                                                        CONDV      11                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  CONDV      12                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  CONDV      13                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  CONDV      14                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          CONDV      15                
      DIMENSION QA(4),QB(13),T(4),THC(4),TSV(5),PV(4),PLV(9),            CONDV      16                
     2         THCD1(5),THCD2(5),THCD3(4),THCD4(9),THCD5(9),             CONDV      17                
     3         TA(5),PB1(5),PB2(5)                                       CONDV      18                
      character cccc*6
      data cccc/5hCONDV/
      DATA   TSV/ 671., 680., 689., 698., 705.47 /                       CONDV      19                
      DATA    PV/ 150., 175., 200., 221.2 /                              CONDV      20                
      DATA   PLV/ 200., 225., 250., 275., 300., 350., 400., 450., 500. / CONDV      21                
      DATA THCD1/ 422.5, 407., 393., 372., 273. /                        CONDV      22                
      DATA THCD2/ 150., 166., 183., 213., 273. /                         CONDV      23                
      DATA THCD3/ 87.5, 106., 126., 270. /                               CONDV      24                
      DATA THCD4/ 416., 433., 450., 461., 472., 491., 507., 523., 536. / CONDV      25                
      DATA THCD5/ 138., 297., 376., 402., 419., 444., 468., 486., 501. / CONDV      26                
      DATA    QA/ -4.51D-8, 1.04D-4, 5.87D-2, 17.6 /                     CONDV      27                
      DATA    QB/ -73.44, 525.77, -1800.7, 2839.5, -922.47, 0.51536,     CONDV      28                
     1-2.0012, 2.5186, -.9473, -7.1693D-4, 2.9323D-3, -3.8929D-3,        CONDV      29                
     2 1.6563D-3 /                                                       CONDV      30                
      DATA    TA/ 400., 425., 450., 500., 550. /                         CONDV      31                
      DATA   PB1/ 175., 225., 275., 350., 450. /                         CONDV      32                
      DATA   PB2/ 170., 220., 270., 345., 445. /                         CONDV      33
C                                                                        STEAMV1M2 - 3/96
      SAVE TSV,PV,PLV,THCD1,THCD2,THCD3,THCD4,THCD5,QA,QB,TA,PB1,PB2     STEAMV1M2 - 3/96
C                                                                        STEAMV1M2 - 2/96
      VAP=1.0                                                            CONDV      34                
      GO TO 10                                                           CONDV      35                
C                                                                        CONDV      36                
      ENTRY CONDL(P,TIN)                                                 STEAMWV1M0(LIB)-NOV. 1,90  13
C                                                                        CONDV      38                
      VAP=0                                                              CONDV      39                
   10 PB=P/14.503773773                                                  CONDV      40                
      IF(P.GT.7500..OR.TIN.GT.TMAX) GO TO 200                            CONDV      41                
      TEMP=TIN                                                           CONDV      42                
      IF(VAP.NE.1.0) GO TO 35                                            CONDV      43                
      IF(PB.GT.221.2) GO TO 35                                           CONDV      44                
      IF(PB.GT.175.0) GO TO 20                                           CONDV      45                
   14 NSET=1                                                             CONDV      46                
      TEMP=TIN                                                           CONDV      47                
C     FUNCTION THCONV                                                    CONDV      48                
   15 TM=(TEMP-32.0)/1.8                                                 CONDV      49                
      H=HSS(P,TEMP,S,VOL)                                                CONDV      50                
      DENS=1./(VOL*62.42796058)                                          CONDV      51                
      C1=0                                                               CONDV      52                
      DO 16 K=1,4                                                        CONDV      53                
   16 C1=C1*TM+QA(K)                                                     CONDV      54                
      DC=((-2.771E-5*TM+0.4198)*TM+103.51)*DENS+                         CONDV      55                
     2  2.1482E14/TM**4.2*DENS**2                                        CONDV      56                
      THCON1=C1+DC                                                       CONDV      57                
      THETA=(TM +273.15)/647.3                                           CONDV      58                
      BETA=PB/100.                                                       CONDV      59                
      A1=(3.08E-4*BETA+3.46E-3)*14.86098521                              CONDV      60                
      B1=1.819700867E5*BETA**1.63/(1+(BETA/2.)**3.26)                    CONDV      61                
      D1=((5.E5*BETA**1.5+2.8E3)/B1-1)*0.206                             CONDV      62                
      THCON2=(A1*THETA**1.445/(1.-B1/ 4.761450262E5/THETA**7)**D1+       CONDV      63                
     2        BETA**4*DEXP(-9.*3.458*(THETA-1.))/(1.+(BETA/2.75)**(-12)) STEAMWV1M0(LIB)-NOV. 1,90  14
     3      *(1.36E-2-3.55E-3*BETA*DEXP(-3.458*(THETA-1.))))*1000.       STEAMWV1M0(LIB)-NOV. 1,90  15
C                                                                        STEAMV1M2 - 2/96
      IF(TEMP.GT.1031.) GO TO 119                                        STEAMV1M2 - 2/96                
      IF(TEMP.LE.662.) GO TO 119                                         STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 2/96
      DO 19 I=1,6                                                        CONDV      68
C                                                                        STEAMV1M2 - 2/96                
      IF(I.EQ.6) GO TO 18                                                CONDV      69                
      IF(TM   .GT.TA(I)) GO TO 19                                        CONDV      70                
      IF(PB.GE.PB1(I)) GO TO 17                                          CONDV      71                
      IF(PB.LE.PB2(I)) GO TO 18                                          CONDV      72                
      R1=(PB1(I)-PB)/5.                                                  CONDV      73                
      R2=(PB-PB2(I))/5.                                                  CONDV      74                
  165 CONDV=R1*THCON1+R2*THCON2                                          CONDV      75                
      GO TO 195                                                          CONDV      76                
   17 CONDV=THCON2                                                       CONDV      77                
      GO TO 195                                                          CONDV      78                
   18 if(I.eq.1) go to 118
      IF(TM   .LE.(TA(I-1)+5.).AND.PB.GE.PB2(I-1)) GO TO 185             CONDV      79                
  118 CONDV=THCON1                                                       CONDV      80                
      GO TO 195                                                          CONDV      81                
  185 R1=(TM -TA(I-1))/5.                                                CONDV      82                
      R2=((TA(I-1)+5.)-TM )/5.                                           CONDV      83                
      GO TO 165                                                          CONDV      84
C                                                                        STEAMV1M2 - 2/96                
   19 CONTINUE                                                           CONDV      85
C                                                                        STEAMV1M2 - 2/96
      GO TO 195                                                          STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 2/96
  119 CONDV=THCON1                                                       STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 2/96                
  195 GO TO (120 ,25,25,25,  90) NSET                                    CONDV      86
C                                                                        STEAMV1M2 - 2/96                
  20  TS=TSL(P)                                                          CONDV      87                
      T(1)=TS                                                            CONDV      88                
      T(2)=734.                                                          CONDV      89                
      T(3)=754.                                                          CONDV      90                
      T(4)=788.                                                          CONDV      91                
      THC(1)=GRS(TSV,1,THCD2,1,T(1),5,NRANGE)                            CONDV      92
C                                                                        STEAMV1M2 - 2/96                
      DO 30 J=2,4                                                        CONDV      93                
      NSET=J                                                             CONDV      94                
      TEMP=T(J)                                                          CONDV      95                
      GO TO 15                                                           CONDV      96                
   25 THC(J)=CONDV                                                       CONDV      97                
   30 CONTINUE                                                           CONDV      98
C                                                                        STEAMV1M2 - 2/96                
      IF(TIN.GT.752.)GO TO 14                                            CONDV      99                
      IF(PB.LE.200..OR. TIN.GE.734.)GO TO 110                            CONDV     100                
      T(4)=T(3)                                                          CONDV     101                
      T(3)=T(2)                                                          CONDV     102                
      T(2)=707.                                                          CONDV     103                
      THC(4)=THC(3)                                                      CONDV     104                
      THC(3)=THC(2)                                                      CONDV     105                
      THC(2)=GRS(PV,1,THCD3,1,PB,4,NRANGE)                               CONDV     106                
      GO TO 110                                                          CONDV     107                
   35 IF(TIN.GT.662.) GO TO 45                                           CONDV     108                
      NRET=1                                                             CONDV     109                
C     FUNCTION THCONL                                                    CONDV     110                
   37 TR=(273.15+(TEMP-32.)/1.8)/273.15                                  CONDV     111                
      PS=PSL(TEMP)                                                       CONDV     112                
      DP=(P-PS)/14.503773773                                             CONDV     113                
      IF(DP.LT.0) DP=0                                                   CONDV     114                
      A=0                                                                CONDV     115                
      B=0                                                                CONDV     116                
      C=0                                                                CONDV     117                
      DO 42 I=1,5                                                        CONDV     118                
   42 A=A*TR+QB(I)                                                       CONDV     119                
      DO 43 I=6,9                                                        CONDV     120                
      B=B*TR+QB(I)                                                       CONDV     121                
   43 C=C*TR+QB(I+4)                                                     CONDV     122                
      CONDV=(C*DP+B)*DP+A                                                CONDV     123                
      THCONL= CONDV                                                      CONDV     124                
      GO TO (120 ,85,92,105) NRET                                        CONDV     125                
  45  T(1)=662.                                                          CONDV     126                
      T(2)=680.                                                          CONDV     127                
      T(3)=707.                                                          CONDV     128                
      T(4)=734.                                                          CONDV     129                
      IF(PB.GT.300.) GO TO 50                                            CONDV     130                
      IF(TIN-734.0) 80,14,14                                             CONDV     131                
   50 IF(PB.LE.400.0) GO TO 70                                           CONDV     132                
      IF(PB.GT.500.0) GO TO 60                                           CONDV     133                
      IF(TIN-797.0) 55,14,14                                             CONDV     134                
   55 T(4)=797.0                                                         CONDV     135                
      GO TO 80                                                           CONDV     136                
   60 IF(TIN-810.0) 65,14,14                                             CONDV     137                
   65 T(4)=810.0                                                         CONDV     138                
      GO TO 80                                                           CONDV     139                
   70 IF(TIN-752.0) 75,14,14                                             CONDV     140                
   75 T(4)=752.0                                                         CONDV     141                
  80  NRET=2                                                             CONDV     142                
      TEMP=T(1)                                                          CONDV     143                
      GO TO 37                                                           CONDV     144                
   85 THC(2)=GRS(PLV,1,THCD4,1,PB,9,NRANGE)                              CONDV     145                
      THC(3)=GRS(PLV,1,THCD5,1,PB,9,NRANGE)                              CONDV     146                
      THC(1)=THCONL                                                      CONDV     147                
      NSET=5                                                             CONDV     148                
      TEMP=T(4)                                                          CONDV     149                
      GO TO 15                                                           CONDV     150                
   90 THC(4)=CONDV                                                       CONDV     151                
      IF(PB.GT.221.2) GO TO 110                                          CONDV     152                
      TS=TSL(P)                                                          CONDV     153                
      T(4)=TS                                                            CONDV     154                
      IF(TS.GT.T(2)) GO TO 95                                            CONDV     155                
      THC(3)=THC(1)                                                      CONDV     156                
      T(3)=T(1)                                                          CONDV     157                
      T(2)=T(3)-20.0                                                     CONDV     158                
      T(1)=T(2)-20.0                                                     CONDV     159                
      NRET=3                                                             CONDV     160                
      TEMP=T(2)                                                          CONDV     161                
      GO TO 37                                                           CONDV     162                
   92 THC(2)=THCONL                                                      CONDV     163                
      GO TO 100                                                          CONDV     164                
   95 T(3)=T(2)                                                          CONDV     165                
      T(2)=T(1)                                                          CONDV     166                
      T(1)=T(1)-20.                                                      CONDV     167                
      THC(3)=THC(2)                                                      CONDV     168                
      THC(2)=THC(1)                                                      CONDV     169                
C     TSL ARRAY EQUALS TSV ARRAY                                         CONDV     170                
  100 THC(4)=GRS(TSV,1,THCD1,1,T(4),5,NRANGE)                            CONDV     171                
      NRET=4                                                             CONDV     172                
      TEMP=T(1)                                                          CONDV     173                
      GO TO 37                                                           CONDV     174                
  200 CALL STER(cccc,12,P,TIN)                                           CONDV     175                
  105 THC(1)=THCONL                                                      CONDV     176                
  110 CONDV =GRS(T,1,THC,1,TIN,4,NRANGE)                                 CONDV     177                
  120 CONDV=CONDV*.5777893E-3                                            CONDV     178                
      RETURN                                                             CONDV     179                
      END                                                                CONDV     180                
