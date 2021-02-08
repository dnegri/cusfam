      FUNCTION P23T(TIN)                                                 P23T        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  33
C  P23T   BOUNDARY BETWEEN REGIONS 2 AND 3                               P23T        3                
C         A 1ST LEVEL SUBROUTINE                                         P23T        4                
C                                                                        P23T        5                
C                                                                        P23T        6                
C                                                                        P23T        7                
C                                                                        P23T        8                
      COMMON /COMCON/ ALPHA0,ALPHA1,   PCA,   VCA,   TCA,   TZA, PV010,  P23T        9                
     1   PVOT,    AI1,    T1,    TC,    P1,  PMIN,  PMAX, PSMAX, P3MIN,  P23T       10                
     2  V3MIN,  V3MAX,  TMIN,  TMAX, TSMAX, T1MAX, T2MIN, T3MIN, T3MAX,  P23T       11                
     3   HMIN,   HMAX, HSMAX, H4MAX,  SMIN,  SMAX, S3MIN, S4MAX          P23T       12                
C                                                                        P23T       13                
      COMMON /CONSTL/ AL0, AL1, AL2, AL2T2                               P23T       14                
C                                                                        P23T       15                
      character cccc*6
      data cccc/5H P23T/
      T = TIN                                                            P23T       16                
      IF(T .LT. TMIN .OR. T .GT. TMAX) CALL STER(cccc, 2, T, T)             P23T       17                
 1000 THETA = (T + TZA)/TCA                                              P23T       18                
      P23T = (AL0 + THETA*(AL1 + THETA*AL2))*PCA                         P23T       19                
 2000 RETURN                                                             P23T       20                
      END                                                                P23T       21                
