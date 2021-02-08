      FUNCTION VPTD(PIN,TIN)                                             VPTD        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  80
CVPTD FUNCTIONS OF PRESSURE AND TEMPERATURE IN DRY REGION                VPTD        3                
C     A 4TH LEVEL SUBROUTINE                                             VPTD        4                
C     WITH ENTRIES VDRY, HPTD, HDRY, SPTD, SDRY                          VPTD        5                
C                                                                        VPTD        6                
C                                                                        VPTD        7                
C     FIRST ARGUMENT IS PRESSURE                                         VPTD        8                
C     SECOND ARGUMENT IS TEMPERATURE                                     VPTD        9                
C     RETURNS WITH SPECIFIC VOLUME, ENTHALPY, OR ENTROPY                 VPTD       10                
C                                                                        VPTD       11                
C     V = F(P,T) IN THE SUPERHEAT OR CRITICAL REGION                     VPTD       12                
C                                                                        VPTD       13                
      P = PIN                                                            VPTD       14                
      T = TIN                                                            VPTD       15                
      P23 = P23T(T)                                                      VPTD       16                
      IF (P .GT. P23) GO TO 1000                                         VPTD       17                
      A = VPT2(P,T)                                                      VPTD       18                
      GO TO 6000                                                         VPTD       19                
C                                                                        VPTD       20                
      ENTRY VDRY(PIN,TIN)                                                STEAMWV1M0(LIB)-NOV. 1,90  81
C                                                                        STEAMV1M2 - 3/96
C     THIS ENTRY CAN BE USED ONLY IF V2E (OR VPT3) WAS LAST CALLED WITH  VPTD       23                
C     THE VALUES OF P (OR V) AND T THAT ARE TO BE ASSUMED HERE.          VPTD       24
C                                                                        STEAMV1M2 - 3/96
      P = PIN                                                            STEAMV1M2 - 3/96
      T = TIN                                                            STEAMV1M2 - 3/96
      P23 = P23T(T)                                                      STEAMV1M2 - 3/96
C                                                                        VPTD       25                
      IF (P .GT. P23) GO TO 1500                                         VPTD       26                
      A = V2E(P,T)                                                         VPTD       27                
      GO TO 6000                                                         VPTD       28                
 1000 V = VPT3D(P,T)                                                     VPTD       29                
 1500 A = V                                                              VPTD       30                
      GO TO 6000                                                         VPTD       31                
C                                                                        VPTD       32                
C     H = F(P,T) IN THE SUPERHEAT OR CRITICAL REGION                     VPTD       33                
C                                                                        VPTD       34                
      ENTRY HPTD(PIN,TIN)                                                STEAMWV1M0(LIB)-NOV. 1,90  82
C     A 4TH LEVEL ENTRY                                                  VPTD       36                
C                                                                        VPTD       37                
      P = PIN                                                            VPTD       38                
      T = TIN                                                            VPTD       39                
      P23 = P23T(T)                                                      VPTD       40                
      IF (P .GT. P23) GO TO 3000                                         VPTD       41                
      A = HPT2(P,T)                                                      VPTD       42                
      GO TO 6000                                                         VPTD       43                
C                                                                        VPTD       44                
      ENTRY HDRY(PIN,TIN)                                                STEAMWV1M0(LIB)-NOV. 1,90  83
C                                                                        VPTD       46                
C     THIS ENTRY CAN BE USED ONLY IF H2E (OR H3E) WAS LAST CALLED WITH   VPTD       47                
C     THE VALUES OF P (OR V) AND T THAT ARE TO BE ASSUMED HERE.          VPTD       48
C                                                                        STEAMV1M2 - 3/96
      P = PIN                                                            STEAMV1M2 - 3/96
      T = TIN                                                            STEAMV1M2 - 3/96
      P23 = P23T(T)                                                      STEAMV1M2 - 3/96
C                                                                        VPTD       49                
      IF (P .GT. P23) GO TO 3500                                         VPTD       50                
      A = H2E(P,T)                                                         VPTD       51                
      GO TO 6000                                                         VPTD       52                
 3000 V = VPT3D(P,T)                                                     VPTD       53                
 3500 A = H3E(V,T)                                                         VPTD       54                
      GO TO 6000                                                         VPTD       55                
C                                                                        VPTD       56                
C     S = F(P,T) IN THE SUPERHEAT OR CRITICAL REGION                     VPTD       57                
C                                                                        VPTD       58                
      ENTRY SPTD(PIN,TIN)                                                STEAMWV1M0(LIB)-NOV. 1,90  84
C     A 4TH LEVEL ENTRY                                                  VPTD       60                
C                                                                        VPTD       61                
      P = PIN                                                            VPTD       62                
      T = TIN                                                            VPTD       63                
      P23 = P23T(T)                                                      VPTD       64                
      IF (P .GT. P23) GO TO 5000                                         VPTD       65                
      A = SPT2(P,T)                                                      VPTD       66                
      GO TO 6000                                                         VPTD       67                
C                                                                        VPTD       68                
      ENTRY SDRY(PIN,TIN)                                                STEAMWV1M0(LIB)-NOV. 1,90  85
C                                                                        VPTD       70                
C     THIS ENTRY CAN BE USED ONLY IF S2E (OR S3E) WAS LAST CALLED WITH   VPTD       71                
C     THE VALUES OF P (OR V) AND T THAT ARE TO BE ASSUMED HERE.          VPTD       72
C                                                                        STEAMV1M2 - 3/96
      P = PIN                                                            STEAMV1M2 - 3/96
      T = TIN                                                            STEAMV1M2 - 3/96
      P23 = P23T(T)                                                      STEAMV1M2 - 3/96
C                                                                        VPTD       73                
      IF (P .GT. P23) GO TO 5500                                         VPTD       74                
      A = S2E(P,T)                                                         VPTD       75                
      GO TO 6000                                                         VPTD       76                
 5000 V = VPT3D(P,T)                                                     VPTD       77                
 5500 A = S3E(V,T)                                                         VPTD       78                
 6000 VPTD = A                                                           VPTD       79                
      RETURN                                                             VPTD       80                
      END                                                                VPTD       81                
