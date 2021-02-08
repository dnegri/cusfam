      SUBROUTINE SATUR (P,T,VF,HF,SF,VG,HG,SG,K)                         SATUR       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  42
C     THIS ROUTINE CALCULATES THE SATURATION PROPERTIES GIVEN EITHER     SATUR       3                
C     SATURATION PRESSURE (K=1) OR TEMPERATURE (K=2) OR BOTH (K=3).      SATUR       4                
C                                                                        SATUR       5                
C                                                                        SATUR       6                
      GO TO (10,20,30),K                                                 SATUR       7                
C     DETERMINE SAT. TEMP.                                               SATUR       8                
   10 T = TSL(P)                                                         SATUR       9                
      GO TO 30                                                           SATUR      10                
C     DETERMINE SAT. PRESSURE.                                           SATUR      11                
   20 T1 = T                                                             SATUR      12                
      P = PSL(T1)                                                        SATUR      13                
C     CHECK LOCATION OF POINT - S.R. 1 AND 2 OR S.R. 3 AND 4             SATUR      14                
   30 IF(T .LE. 662.0) GO TO 40                                          SATUR      15                
C     SUBREGION 3 AND 4 EQUATIONS WILL BE USED TO CALC. SAT. PROPERTIES. SATUR      16                
      VF = VPTF3(P,T)                                                    SATUR      17                
      HF=HVT3(VF,T)                                                      SATUR      18                
      SF = S3E(VF,T)                                                      SATUR      19                
      VG = VPTG3(P,T)                                                    SATUR      20                
      HG=HVT3(VG,T)                                                      SATUR      21                
      SG = S3E(VG,T)                                                      SATUR      22                
      RETURN                                                             SATUR      23                
C                                                                        SATUR      24                
C     SUBREGION 1 AND 2 EQUATIONS WILL BE USED TO CALC. SAT. PROPERTIES. SATUR      25                
   40 VF = VPT1(P,T)                                                     SATUR      26                
      HF = H1E(P,T)                                                      SATUR      27                
      SF = S1E(P,T)                                                      SATUR      28                
      VG = VPT2(P,T)                                                     SATUR      29                
      HG = H2E(P,T)                                                      SATUR      30                
      SG = S2E(P,T)                                                      SATUR      31                
      RETURN                                                             SATUR      32                
      END                                                                SATUR      33                
