      FUNCTION HPTL(PIN,TIN)                                             HPTL        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  29
CHPTL H = F(P,T) IN LIQUID REGION                                        HPTL        3                
C     A 4TH LEVEL SUBROUTINE                                             HPTL        4                
C                                                                        HPTL        5                
C                                                                        HPTL        6                
C     FIRST ARGUMENT IS PRESSURE                                         HPTL        7                
C     SECOND ARGUMENT IS TEMPERATURE                                     HPTL        8                
C     RETURNS WITH SPECIFIC ENTHALPY                                     HPTL        9                
C                                                                        HPTL       10                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  HPTL       11                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  HPTL       12                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  HPTL       13                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          HPTL       14                
C                                                                        HPTL       15                
      P = PIN                                                            HPTL       16                
      T = TIN                                                            HPTL       17                
      IF ( T .LT.T1) GO TO 1000                                          HPTL       18                
      V = VPT3L(P,T)                                                     HPTL       19                
      HPTL = H3E(V,T)                                                      HPTL       20                
      GO TO 2000                                                         HPTL       21                
 1000 HPTL = HPT1(P,T)                                                   HPTL       22                
 2000 RETURN                                                             HPTL       23                
      END                                                                HPTL       24                
