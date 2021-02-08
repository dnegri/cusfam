      FUNCTION VISL (P,TIN)                                              VISL        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  61
C     VISL    VISCOSITY AS A FUNCTION OF PRESSURE AND TEMPERATURE        VISL        3                
C          A 6TH LEVEL SUBROUTINE                                        VISL        4                
C                                                                        VISL        5                
C                                                                        VISL        6                
C     FIRST ARGUMENT IS PRESSURE                                         VISL        7                
C     SECOND ARGUMENT IS TEMPERATURE                                     VISL        8                
C     RETURNS WITH VISCOSITY IN LB/FT SEC                                VISL        9                
C     VISV IS ENTRY FOR VAPOR REGION                                     VISL       10                
C     VISL IS ENTRY FOR LIQUID REGION                                    VISL       11                
C                                                                        VISL       12                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  VISL       13                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  VISL       14                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  VISL       15                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          VISL       16                
      DIMENSION T(4), VIS(4), PA(16), VA(16), VB(16), VC(10), VD(8), TSC VISL       17                
     1(6),VE(6)                                                          VISL       18                
      character cccc*6
      data cccc/4hVISL/
      DATA PA /   0., 200., 225., 250., 275., 300., 350., 400., 450.,    VISL       19                
     1          500., 550., 600., 650., 700., 750., 800./                VISL       20                
      DATA VA / 822., 843., 850., 856., 863., 869., 883., 897., 911.,    VISL       21                
     1          925., 939., 953., 966., 979., 992.,1004./                VISL       22                
      DATA VB / 716., 733., 746., 759., 772., 783., 801., 818., 834.,    VISL       23                
     1          849., 864., 879., 893., 907., 921.,934./                 VISL       24                
      DATA VC / 695., 724., 748., 769., 788., 806., 823., 839., 854.,    VISL       25                
     1          868./                                                    VISL       26                
      DATA VD / 665., 695., 718., 739., 758., 775., 790., 804./          VISL       27                
      DATA TSC/ 300., 325., 350., 360., 370., 374.15 /                   VISL       28                
      DATA VE / 900., 822., 716., 662., 565., 433.5  /                   VISL       29
C                                                                        STEAMV1M2 - 3/96
      SAVE PA,VA,VB,VC,VD,TSC,VE                                         STEAMV1M2 - 3/96
C                                                                        VISL       30                
      NV=0                                                               VISL       31                
      GO TO 5                                                            VISL       32                
C                                                                        VISL       33                
      ENTRY VISV(P,TIN)                                                  STEAMWV1M0(LIB)-NOV. 1,90  62
C     A 6TH LEVEL ENTRY                                                  VISL       35                
C                                                                        VISL       36                
      NV=1                                                               VISL       37                
    5 NRET=1                                                             VISL       38                
      TEMP=TIN                                                           VISL       39                
      IF (P.GT.12500..OR.TIN.GT.TMAX) GO TO 100                          VISL       40                
      PB=P/14.503773773                                                  VISL       41                
      TB=(TIN-32.)/1.8                                                   VISL       42                
      IF (P.GT.PCA) GO TO 20                                             VISL       43                
      IF (NV) 10,20,10                                                   VISL       44                
   10 DENS=HSS(P,TEMP,DENS,VOL)                                          VISL       45                
      DENS=0.0160184633/VOL                                              VISL       46                
      TM=(TEMP-32.)/1.8                                                  VISL       47                
      DV1=-DENS*(1858.0-5.9*TM)                                          VISL       48                
      DV=((102.1*DENS+676.5)*DENS+353.0)*DENS                            STEAMWV1M0(LIB)-NOV. 1,90  63
      DV2=DV                                                             STEAMWV1M0(LIB)-NOV. 1,90  64
      IF (TM.GE.365.0) GO TO 15                                          VISL       50                
      DV=DV1                                                             VISL       51                
      IF (TM.LE.340.0) GO TO 15                                          VISL       52                
      DV=(DV1-DV2)*(365.0-TM)/25.+DV2                                    VISL       53                
   15 V=0.407*TM+80.4+DV                                                 STEAMWV1M0(LIB)-NOV. 1,90  65
      VIS(4)=V                                                           STEAMWV1M0(LIB)-NOV. 1,90  66
      GO TO (95,90), NRET                                                VISL       55                
   20 NSET=1                                                             VISL       56                
      IF (TIN.LT.572.0) GO TO 45                                         VISL       57                
      IF (TIN-797.) 30,25,25                                             VISL       58                
   25 TEMP=TIN                                                           VISL       59                
      NRET=1                                                             VISL       60                
      GO TO 10                                                           VISL       61                
   30 IF (TIN.LT.752.) GO TO 35                                          VISL       62                
      IF (PB-450.) 25,40,40                                              VISL       63                
   35 IF (TIN.LT.707.) GO TO 40                                          VISL       64                
      IF (PB.LE.350.) GO TO 25                                           VISL       65                
   40 T(1)=300.0                                                         VISL       66                
      T(2)=325.0                                                         VISL       67                
      T(3)=350.0                                                         VISL       68                
      T(4)=375.0                                                         VISL       69                
      NSET=2                                                             VISL       70                
      TEMP=572.0                                                         VISL       71                
   45 TK=273.15+(TEMP-32.)/1.8                                           VISL       72                
      DP=(P-PSL(TEMP))/14.503773773D6                                    STEAMV1M2 - 2/96             
      IF (DP.LT.0) DP=0                                                  VISL       74                
      PHI=1.04673*(TK-305.0)                                             VISL       75                
      Y=247.8/(TK-140.0)                                                 VISL       76                
      VIS(1)=241.4*10.**Y*(1.+(DP*PHI))                                  STEAMWV1M0(LIB)-NOV. 1,90  67
      V=VIS(1)                                                           STEAMWV1M0(LIB)-NOV. 1,90  68
      GO TO (95,50,60,90), NSET                                          VISL       78                
   50 PA(1)=120.5569                                                     VISL       79                
      VIS(2)=GRS(PA,1,VA,1,PB,16,NRANGE)                                 VISL       80                
      PA(1)=165.35125                                                    VISL       81                
      VIS(3)=GRS(PA,1,VB,1,PB,16,NRANGE)                                 VISL       82                
      VIS(4)=GRS(PA(7),1,VC,1,PB,10,NRANGE)                              VISL       83                
      IF (TB.GT.374.15) GO TO 65                                         VISL       84                
      IF (P.GT.PCA) GO TO 65                                             VISL       85                
      T(4)=(TSL(P)-32.)/1.8                                              VISL       86                
      VIS(4)=GRS(TSC,1,VE,1,T(4),6,NRANGE)                               VISL       87                
      IF (T(4).GT.350.0) GO TO 90                                        VISL       88                
      NSET=3                                                             VISL       89                
      TEMP=562.0                                                         VISL       90                
   55 VIS(3)=VIS(2)                                                      VISL       91                
      VIS(2)=VIS(1)                                                      VISL       92                
      T(3)=T(2)                                                          VISL       93                
      T(2)=T(1)                                                          VISL       94                
      T(1)=(TEMP-32.)/1.8                                                VISL       95                
      GO TO 45                                                           VISL       96                
   60 IF (T(4).GT.325.0) GO TO 90                                        VISL       97                
      NSET=4                                                             VISL       98                
      TEMP=552.0                                                         VISL       99                
      GO TO 55                                                           VISL      100                
   65 IF (TB.LT.375.) GO TO 90                                           VISL      101                
      NRET=2                                                             VISL      102                
      NREX=1                                                             VISL      103                
   70 VIS(1)=VIS(2)                                                      VISL      104                
      VIS(2)=VIS(3)                                                      VISL      105                
      VIS(3)=VIS(4)                                                      VISL      106                
      T(1)=T(2)                                                          VISL      107                
      T(2)=T(3)                                                          VISL      108                
      T(3)=T(4)                                                          VISL      109                
      GO TO (75,85), NREX                                                VISL      110                
   75 TEMP=752.                                                          VISL      111                
      T(4)=400.                                                          VISL      112                
      IF (PB.LT.450.) GO TO 10                                           VISL      113                
      VIS(4)=GRS(PA(9),1,VD,1,PB,8,NRANGE)                               VISL      114                
      GO TO (80,90), NREX                                                VISL      115                
   80 NREX=2.                                                            VISL      116                
      GO TO 70                                                           VISL      117                
   85 TEMP=797.                                                          VISL      118                
      PA(1)=0.                                                           VISL      119                
      T(4)=425.                                                          VISL      120                
      GO TO 10                                                           VISL      121                
   90 V=GRS(T,1,VIS,1,TB,4,NRANGE)                                       VISL      122                
   95 VISL=0.67197D-7*V                                                  STEAMV1M2 - 2/96             
      RETURN                                                             VISL      124                
  100 CALL STER (cccc,12,P,TIN)                                          VISL      125                
C     END OF VISL                                                        VISL      126                
      END                                                                VISL      127                
