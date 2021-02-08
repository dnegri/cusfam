      SUBROUTINE SRSORT (P, T, V, H, S, ISAT, VG, HG, SG)                SRSORT      2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  50
C     THIS ROUTINE RECIEVES A TEMPERATURE AND PRESSURE AND DETERMINES    SRSORT      3                
C     WHICH SUBREGION (1,2,3,OR 4) THIS POINT LIES IN.  IT THEN          SRSORT      4                
C     CALCULATES SPECIFIC VOLUME (V), ENTHALPY (H), AND ENTROPY (S) FOR  SRSORT      5                
C     THIS POINT.  IF THE POINT IS AT SATURATION, SWITCH ISAT IS SET TO  SRSORT      6                
C     A VALUE OF 2 AND VF, HF, AND SF ARE RETURNED IN V, H, AND S AND    SRSORT      7                
C     THE VAPOR SIDE PROPERTIES ARE RETURNED AS VG, HG, AND SG.          SRSORT      8                
C                                                                        SRSORT      9                
C                                                                        SRSORT     10                
      ISAT = 1                                                           SRSORT     11                
C     CHECK IF POINT LIES ABOVE THE SATURATION DOME, IF SO GO TO S.R. 2  SRSORT     12                
      IF(T .GT. 705.47) GO TO 10                                         SRSORT     13                
C     DETERMINE THE SATURATION TEMPERATURE.                              SRSORT     14                
      T1 = T                                                             SRSORT     15                
      PSAT = PSL(T1)                                                     SRSORT     16                
C     IF AT SATURATION, GO TO CALC. SAT. PROPERTIES.                     SRSORT     17                
      IF(P .EQ. PSAT) GO TO 60                                           SRSORT     18                
C     IF T IS BELOW 662. F THEN POINT MUST BE IN S.R. 1 OR 2.            SRSORT     19                
      IF(T .LE. 662.0) GO TO 30                                          SRSORT     20                
C     CHECK IF IN SUBREGION 4.                                           SRSORT     21                
      IF(P .GT. PSAT) GO TO 52                                           SRSORT     22                
C     THE FOLLOWING IS FOR SUBREGIONS 2 OR 3.                            SRSORT     23                
   10 P23 = P23T(T)                                                      SRSORT     24                
      IF(P .GT. P23) GO TO 50                                            SRSORT     25                
C     THE FOLLOWING IS FOR SUBREGION 2 ONLY.                             SRSORT     26                
   20 V = VPT2(P,T)                                                      SRSORT     27                
      H = H2E(P,T)                                                       SRSORT     28                
      S = S2E(P,T)                                                       SRSORT     29                
      RETURN                                                             SRSORT     30                
C     THE FOLLOWING CHECKS FOR SUBREGION 1 OR 2.                         SRSORT     31                
   30 IF(T .LT. 25.0) GO TO 70                                           SRSORT     32                
      IF(P .LT. PSAT) GO TO 20                                           SRSORT     33                
C     THE FOLLOWING IS FOR SUBREGION 1 ONLY.                             SRSORT     34                
   40 V = VPT1(P,T)                                                      SRSORT     35                
      H = H1E (P,T)                                                      SRSORT     36                
      S = S1E (P,T)                                                      SRSORT     37                
      RETURN                                                             SRSORT     38                
C     THE FOLLOWING IS FOR SUBREGIONS 3 AND 4.                           SRSORT     39                
   50 V = VPT3D(P,T)                                                     SRSORT     40                
  51  H=HVT3(V,T)                                                        SRSORT     41                
      S = S3E(V,T)                                                       SRSORT     42                
      RETURN                                                             SRSORT     43                
  52  V = VPT3L(P,T)                                                     SRSORT     44                
      GO TO 51                                                           SRSORT     45                
C     THE FOLLOWING IS FOR SATURATION PROPERTIES.                        SRSORT     46                
   60 ISAT = 2                                                           SRSORT     47                
      K = 3                                                              SRSORT     48                
      CALL SATUR (P,T,V,H,S,VG,HG,SG,K)                                  SRSORT     49                
      RETURN                                                             SRSORT     50                
C     THE FOLLOWING IS FOR INDICATING ERRORS.                            SRSORT     51                
   70 ISAT = 3                                                           SRSORT     52                
      RETURN                                                             SRSORT     53                
      END                                                                SRSORT     54                
