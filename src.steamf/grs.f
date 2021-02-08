           FUNCTION GRS   (X, NDX, Y, NDY, XV, N, NRANGE)                GRS         2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  23
C  GRS    INTERPOLATION SUBROUTINE                                       GRS         3                
C                                                                        GRS         4                
C                                                                        GRS         5                
C     WHERE                                                              GRS         6                
C     X = ARRAY OF POINTS ON ABSCISSA                                    GRS         7                
C     NDX = INCREMENT BETWEEN POINTS IN THE X ARRAY                      GRS         8                
C     Y = ARRAY OF POINTS ON ORDINATE                                    GRS         9                
C     NDY = INCREMENT BETWEEN POINTS IN THE Y ARRAY                      GRS        10                
C     XV = KNOWN VALUE OF X                                              GRS        11                
C     N = NUMBER OF POINTS IN TABLE                                      GRS        12                
C     NRANGE = EXTRAPOLATION INDICATOR                                   GRS        13                
C                                                                        GRS        14                
      DIMENSION X(NDX,16), Y(NDY,16), DX(16), DY(16), YP(16)                  GRS        15                
      NRANGE = 0                                                         GRS        16                
      IF(XV .LT. X(1,1)) NRANGE = -1                                     GRS        17                
      IF(XV .GT. X(1,N)) NRANGE = +1                                     GRS        18                
      DO 10 I = 1,N                                                      GRS        19                
      IF(XV - X(1,I)) 20,70,10                                           GRS        20                
   10 CONTINUE                                                           GRS        21                
      I = N                                                              GRS        22                
   20 IF(I .GT. 2) GO TO 30                                              GRS        23                
      N1 = 3                                                             GRS        24                
      N2 = 2                                                             GRS        25                
      N3 = 1                                                             GRS        26                
      NP = 3                                                             GRS        27                
      GO TO 55                                                           GRS        28                
   30 IF(I .LT. N) GO TO 40                                              GRS        29                
      NP = 3                                                             GRS        30                
      GO TO 50                                                           GRS        31                
   40 NP = 4                                                             GRS        32                
      N4 = I + 1                                                         GRS        33                
   50 N1 = I - 2                                                         GRS        34                
      N2 = I - 1                                                         GRS        35                
      N3 = I                                                             GRS        36                
   55 DX(1) =  X(1,N2) - X(1,N1)                                         GRS        37                
      DY(1) =  Y(1,N2) - Y(1,N1)                                         GRS        38                
      DX(2) =  X(1,N3) - X(1,N2)                                         GRS        39                
      DY(2) =  Y(1,N3) - Y(1,N2)                                         GRS        40                
      R = (XV - X(1,N2))/DX(2)                                           GRS        41                
      YP(1) = (DY(1)*DX(2)**2 + DY(2)*DX(1)**2)/(DX(1)*(DX(1) + DX(2)))  GRS        42                
      IF(NP .EQ. 4) GO TO 60                                             GRS        43                
      GRS = Y(1,N2) + R*(YP(1) + R*(DY(2) - YP(1)))                      GRS        44                
      GO TO 9999                                                         GRS        45                
   60 DX(3) = X(1,N4) - X(1,N3)                                          GRS        46                
      DY(3) = Y(1,N4) - Y(1,N3)                                          GRS        47                
      YP(2) = (DY(2)*DX(3)**2 + DY(3)*DX(2)**2)/(DX(3)*(DX(2) + DX(3)))  GRS        48                
      GRS = Y(1,N2) + R*(YP(1) + R*(3.*DY(2) - 2.*YP(1) - YP(2) +        GRS        49                
     1 R*(YP(1) + YP(2) - 2.*DY(2))))                                    GRS        50                
      GO TO 9999                                                         GRS        51                
   70 GRS = Y(1,I)                                                       GRS        52                
 9999 RETURN                                                             GRS        53                
      END                                                                GRS        54                
