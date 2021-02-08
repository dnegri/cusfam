      FUNCTION SIGMA(T)                                                  SIGMA       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  43
      DIMENSION A(5)                                                     SIGMA       3                
      DATA A/.1160936807,.001121404688,-5.75280518D-06,1.28627465D-08,   STEAMV1M2 - 2/96
     1-1.14971929D-11/                                                   STEAMV1M2 - 2/96
C                                                                        STEAMV1M2 - 3/96
      SAVE A                                                             STEAMV1M2 - 3/96
C                                                                        STEAMV1M2 - 3/96
      SUM = 0.                                                           SIGMA       6                
      BETA = 0.83                                                        SIGMA       7                
      TF = 391.93 - T / 1.8                                              SIGMA       8                
      DO 5 N = 2,5                                                       SIGMA       9                
      SUM = SUM + A(N) * TF ** N                                         SIGMA      10                
    5 CONTINUE                                                           SIGMA      11                
      SIGMA = .00006852*(A(1) * TF **2 / ( 1. + BETA * TF ) + SUM )      SIGMA      12                
      RETURN                                                             SIGMA      13                
      END                                                                SIGMA      14                
