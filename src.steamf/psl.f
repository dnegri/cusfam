      FUNCTION PSL(T)                                                    PSL         2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  36
C  PSL    SATURATION PRESSURE AS A FUNCTION OF TEMPERATURE.              PSL         3                
C         ENTRY = PSATT(DUMMY)                                           PSL         4                
C                                                                        PSL         5                
C                                                                        PSL         6                
C                                                                        PSL         7                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  PSL         8                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  PSL         9                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  PSL        10                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          PSL        11                
      COMMON /NUST/ DPDT                                                 PSL        12                
      DIMENSION AK(9)                                                    PSL        13                
      character cccc*6
      data cccc/3hPSL/
      DATA    AK    /-7.691234564,-26.08023696,-168.1706546,64.23285504, PSL        14                
     1  -118.9646225, 4.167117320, 20.97506760,1.0D+9,6.0/               STEAMV1M2 - 2/96             
      DATA AK2T2,AK3T3/-5.21604739D+1,-5.04511964D+2/                    STEAMV1M2 - 2/96             
      DATA AK4T4,AK5T5/+2.56931420D+2,-5.94823113D+2/                    STEAMV1M2 - 2/96             
      DATA AK7T2,AK8T2/+4.19501352D+1,+2.0D+9/                           STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE AK,AK2T2,AK3T3,AK4T4,AK5T5,AK7T2,AK8T2                        STEAMV1M2 - 3/96
C                                                                        PSL        19                
      ENTRY PSATT(T)                                                     STEAMWV1M0(LIB)-NOV. 1,90  37
C                                                                        PSL        21                
      J=1                                                                PSL        22                
      GO TO 5                                                            PSL        23                
C                                                                        PSL        24                
      ENTRY PSL1(T)                                                      STEAMWV1M0(LIB)-NOV. 1,90  38
C                                                                        PSL        26                
      J=2                                                                PSL        27                
  5   IF(T.GT.TC) GO TO 15                                               PSL        28                
C     THETA = ((T-32.0)/1.8 + 273.15)/647.3                              PSL        29                
      THETA=(T+TZA)/TCA                                                  PSL        30                
      X = 1.0- THETA                                                     PSL        31                
      Y=0.                                                               PSL        32                
      DO 10 I=1,5                                                        PSL        33                
      II=6-I                                                             PSL        34                
  10  Y=(Y+AK(II))*X                                                     PSL        35                
C     THE K FUNCTION (SATURATION LINE) PAGE 12 PAR. 5                    PSL        36                
      DEN1=1.+X*(AK(6)+AK(7)*X)                                          PSL        37                
      DEN2=AK(8)*X*X+AK(9)                                               PSL        38                
      PSL=DEXP(Y/THETA/DEN1-X/DEN2)*PCA                                  STEAMWV1M0(LIB)-NOV. 1,90  39
      IF(J.NE.2) GO TO 20                                                PSL        40                
      DSDT=-(AK(1)+X*(AK2T2+X*(AK3T3+X*(AK4T4+AK5T5*X))))                PSL        41                
      B=THETA*DEN1                                                       PSL        42                
      DBDT=DEN1-THETA*(AK(6)+AK7T2*X)                                    PSL        43                
      DBBDT=-AK8T2*X                                                     PSL        44                
      DPDT=(PSL/1165.14)*(((B*DSDT-Y*DBDT)/(B*B))+((DEN2+X*DBBDT)/(DEN2* PSL        45                
     1DEN2)))                                                            PSL        46                
      GO TO 20                                                           PSL        47                
  15  CALL STER(cccc,2,T,0.d0)                                              PSL        48                
  20  RETURN                                                             PSL        49                
C     END OF PSL                                                         PSL        50                
      END                                                                PSL        51                
