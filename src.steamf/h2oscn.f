      SUBROUTINE H2OSCN (IOPT, XIND, PRESS, TEMP, ENTH, VOL, ENTR, TOL ) H2OSCN      2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  24
      JS = 1                                                             H2OSCN      3                
      IF(IOPT .GT. 3 ) GO TO 100                                         H2OSCN      4                
      KODE = 1                                                           H2OSCN      5                
      PA = PRESS                                                         H2OSCN      6                
      GO TO 101                                                          H2OSCN      7                
 100  KODE = 2                                                           H2OSCN      8                
      TA = TEMP                                                          H2OSCN      9                
 101  CALL SATUR( PA, TA, VF, HF, SF, VG, HG, SG, KODE )                 H2OSCN     10                
      GO TO ( 1,2,3,4,5,6 ),IOPT                                         H2OSCN     11                
   1  IF( XIND .GT. HF ) GO TO 14                                        H2OSCN     12                
      JQ = 1                                                             H2OSCN     13                
      T = TA - 10.                                                       H2OSCN     14                
      HI = HF                                                            H2OSCN     15                
  10  TI = TA                                                            H2OSCN     16                
      HLO = XIND * ( 1. - TOL )                                          H2OSCN     17                
      HHI = XIND * ( 1. + TOL )                                          H2OSCN     18                
      DO 7  I = 1,25                                                     H2OSCN     19                
      CALL SRSORT ( PRESS, T, V, H, S, ISAT, VG, HG, SG )                H2OSCN     20                
      IF ( H .GT. HLO .AND.H .LT. HHI ) GO TO 8                          H2OSCN     21                
      TX = T - ( TI - T ) / ( HI - H ) * ( H - XIND )                    H2OSCN     22                
      IF ( JQ .EQ. 2 ) GO TO 12                                          H2OSCN     23                
      IF ( TX .GT. TA )  TX = TA                                         H2OSCN     24                
      GO TO 13                                                           H2OSCN     25                
  12  IF ( TX .LT. TA ) TX = TA                                          H2OSCN     26                
  13  HI = H                                                             H2OSCN     27                
      TI = T                                                             H2OSCN     28                
   7  T  = TX                                                            H2OSCN     29                
  199 WRITE ( 6,9 ) IOPT, PRESS, XIND                                    H2OSCN     30                
  9   FORMAT(27H0FAIL TO CONVERGE - OPTION ,I2,14H - PRESSURE = ,F9.3,   H2OSCN     31                
     219H PSIA - ENTHALPY = ,F9.3,9H BTU/LBM  )                          H2OSCN     32                
      JS = 2                                                             H2OSCN     33                
      GO TO 200                                                          H2OSCN     34                
  14  IF ( XIND .LT. HG ) GO TO 11                                       H2OSCN     35                
      T = TA + 10.                                                       H2OSCN     36                
      HI = HG                                                            H2OSCN     37                
      JQ = 2                                                             H2OSCN     38                
      GO TO 10                                                           H2OSCN     39                
  11  TEMP = TA                                                          H2OSCN     40                
      ENTH = XIND                                                        H2OSCN     41                
      X = (XIND - HF ) / ( HG - HF )                                     H2OSCN     42                
      VOL = VF  + X * (VG - VF )                                         H2OSCN     43                
      ENTR = SF + X * (SG - SF )                                         H2OSCN     44                
      GO TO 200                                                          H2OSCN     45                
   8  TEMP = T                                                           H2OSCN     46                
  15  ENTH = H                                                           H2OSCN     47                
      VOL  = V                                                           H2OSCN     48                
      ENTR = S                                                           H2OSCN     49                
      GO TO 200                                                          H2OSCN     50                
    2 IF( XIND.GT.VF ) GO TO 24                                          H2OSCN     51                
      JQ = 1                                                             H2OSCN     52                
      T = TA - 10.                                                       H2OSCN     53                
      VI = VF                                                            H2OSCN     54                
   20 TI = TA                                                            H2OSCN     55                
      VLO = XIND * ( 1. - TOL )                                          H2OSCN     56                
      VHI = XIND * ( 1. + TOL )                                          H2OSCN     57                
      DO 27 I = 1,25                                                     H2OSCN     58                
      CALL SRSORT (PRESS,T,V,H,S,ISAT,VG,HG,SG )                         H2OSCN     59                
      IF ( V.GT. VLO .AND. V.LT. VHI ) GO TO 8                           H2OSCN     60                
      TX = T -(TI-T) / (VI-V) * (V- XIND )                               H2OSCN     61                
      IF ( JQ.EQ.2) GO TO 22                                             H2OSCN     62                
      IF ( TX.GT.TA ) TX = TA                                            H2OSCN     63                
      GO TO 23                                                           H2OSCN     64                
   22 IF ( TX .LT. TA ) TX = TA                                          H2OSCN     65                
   23 VI = V                                                             H2OSCN     66                
      TI = T                                                             H2OSCN     67                
   27 T = TX                                                             H2OSCN     68                
      GO TO 199                                                          H2OSCN     69                
   24 IF ( XIND .LT. VG ) GO TO 21                                       H2OSCN     70                
      T= TA + 10.                                                        H2OSCN     71                
      VI = VG                                                            H2OSCN     72                
      JQ = 2                                                             H2OSCN     73                
      GO TO 20                                                           H2OSCN     74                
   21 TEMP = TA                                                          H2OSCN     75                
      VOL = XIND                                                         H2OSCN     76                
      X = ( XIND - VF ) / ( VG -VF )                                     H2OSCN     77                
      ENTH = HF + X * (HG - HF )                                         H2OSCN     78                
      ENTR = SF + X * (SG - SF )                                         H2OSCN     79                
      GO TO 200                                                          H2OSCN     80                
    3 IF( XIND.GT.SF ) GO TO 34                                          H2OSCN     81                
      JQ = 1                                                             H2OSCN     82                
      T = TA - 10.                                                       H2OSCN     83                
      SI = SF                                                            H2OSCN     84                
   30 TI = TA                                                            H2OSCN     85                
      SLO = XIND * ( 1. - TOL )                                          H2OSCN     86                
      SHI = XIND * ( 1. + TOL )                                          H2OSCN     87                
      DO 37 I = 1,25                                                     H2OSCN     88                
      CALL SRSORT (PRESS,T,V,H,S,ISAT,VG,HG,SG )                         H2OSCN     89                
      IF ( S.GT. SLO .AND. S.LT. SHI ) GO TO 8                           H2OSCN     90                
      TX= T -(TI-T) / (SI-S) * (S- XIND )                                H2OSCN     91                
      IF ( JQ.EQ.2) GO TO 32                                             H2OSCN     92                
      IF ( TX.GT.TA ) TX = TA                                            H2OSCN     93                
      GO TO 33                                                           H2OSCN     94                
   32 IF ( TX .LT. TA ) TX = TA                                          H2OSCN     95                
   33 SI = S                                                             H2OSCN     96                
      TI = T                                                             H2OSCN     97                
   37 T  = TX                                                            H2OSCN     98                
      GO TO 199                                                          H2OSCN     99                
   34 IF ( XIND .LT. SG ) GO TO 31                                       H2OSCN    100                
      T= TA + 10.                                                        H2OSCN    101                
      SI = SG                                                            H2OSCN    102                
      JQ = 2                                                             H2OSCN    103                
      GO TO 30                                                           H2OSCN    104                
   31 TEMP = TA                                                          H2OSCN    105                
      ENTR= XIND                                                         H2OSCN    106                
      X = ( XIND - SF ) / ( SG -SF )                                     H2OSCN    107                
      ENTH = HF + X * (HG - HF )                                         H2OSCN    108                
      VOL  = VF + X * (VG - VF )                                         H2OSCN    109                
      GO TO 200                                                          H2OSCN    110                
    4 IF ( XIND.GT.HF ) GO TO 44                                         H2OSCN    111                
      JQ = 1                                                             H2OSCN    112                
      P = PA + 100.                                                      H2OSCN    113                
      HI = HF                                                            H2OSCN    114                
   40 PI = PA                                                            H2OSCN    115                
      HLO = XIND * ( 1. - TOL )                                          H2OSCN    116                
      HHI = XIND * ( 1. + TOL )                                          H2OSCN    117                
      DO 47 I = 1,25                                                     H2OSCN    118                
      CALL SRSORT (P,TEMP,V,H,S,ISAT,VG,HG,SG )                          H2OSCN    119                
      IF ( H.GT. HLO .AND. H.LT. HHI ) GO TO 45                          H2OSCN    120                
      PX= P -(PI-P) / (HI-H) * (H- XIND )                                H2OSCN    121                
      IF ( JQ.EQ.2) GO TO 42                                             H2OSCN    122                
      IF ( PX.LT.PA ) PX = PA* ( 1. + .05/ I )                           H2OSCN    123                
      GO TO 43                                                           H2OSCN    124                
   42 IF ( PX .GT. PA ) PX = PA * ( 1. - .05 / I )                       H2OSCN    125                
   43 HI = H                                                             H2OSCN    126                
      PI = P                                                             H2OSCN    127                
   47 P  = PX                                                            H2OSCN    128                
      GO TO 199                                                          H2OSCN    129                
   44 IF ( XIND .LT. HG ) GO TO 41                                       H2OSCN    130                
      P= PA - 50.                                                        H2OSCN    131                
      HI = HG                                                            H2OSCN    132                
      JQ = 2                                                             H2OSCN    133                
      GO TO 40                                                           H2OSCN    134                
   41 PRESS= PA                                                          H2OSCN    135                
      ENTH= XIND                                                         H2OSCN    136                
      X = ( XIND - HF ) / ( HG -HF )                                     H2OSCN    137                
      VOL  = VF + X * (VG - VF )                                         H2OSCN    138                
      ENTR = SF + X * (SG - SF )                                         H2OSCN    139                
      GO TO 200                                                          H2OSCN    140                
  45  PRESS = P                                                          H2OSCN    141                
      GO TO 15                                                           H2OSCN    142                
    5 IF ( XIND.GT.VF ) GO TO 54                                         H2OSCN    143                
      JQ = 1                                                             H2OSCN    144                
      P = PA + 100.                                                      H2OSCN    145                
      VI = VF                                                            H2OSCN    146                
   50 PI = PA                                                            H2OSCN    147                
      VLO = XIND * ( 1. - TOL )                                          H2OSCN    148                
      VHI = XIND * ( 1. + TOL )                                          H2OSCN    149                
      DO 57 I = 1,25                                                     H2OSCN    150                
      CALL SRSORT (P,TEMP,V,H,S,ISAT,VG,HG,SG )                          H2OSCN    151                
      IF ( V.GT. VLO .AND. V.LT. VHI ) GO TO 45                          H2OSCN    152                
      PX= P -(PI-P) / (VI-V) * (V- XIND )                                H2OSCN    153                
      IF ( JQ.EQ.2) GO TO 52                                             H2OSCN    154                
      IF ( PX.LT.PA ) PX = PA * ( 1. + .05 / I )                         H2OSCN    155                
      GO TO 53                                                           H2OSCN    156                
   52 IF ( PX .GT. PA ) PX = PA * ( 1. - .05 / I )                       H2OSCN    157                
   53 VI = V                                                             H2OSCN    158                
      PI = P                                                             H2OSCN    159                
   57 P  = PX                                                            H2OSCN    160                
      GO TO 199                                                          H2OSCN    161                
   54 IF ( XIND .LT. VG ) GO TO 51                                       H2OSCN    162                
      P= PA - 50.                                                        H2OSCN    163                
      VI = VG                                                            H2OSCN    164                
      JQ = 2                                                             H2OSCN    165                
      GO TO 50                                                           H2OSCN    166                
   51 PRESS= PA                                                          H2OSCN    167                
      VOL = XIND                                                         H2OSCN    168                
      X = ( XIND - VF ) / ( VG -VF )                                     H2OSCN    169                
      ENTH = HF + X * (HG - HF )                                         H2OSCN    170                
      ENTR = SF + X * (SG - SF )                                         H2OSCN    171                
      GO TO 200                                                          H2OSCN    172                
    6 IF( XIND.GT.SF ) GO TO 64                                          H2OSCN    173                
      JQ = 1                                                             H2OSCN    174                
      P = PA + 100.                                                      H2OSCN    175                
      SI = SF                                                            H2OSCN    176                
   60 PI = PA                                                            H2OSCN    177                
      SLO = XIND * ( 1. - TOL )                                          H2OSCN    178                
      SHI = XIND * ( 1. + TOL )                                          H2OSCN    179                
      DO 67 I = 1,25                                                     H2OSCN    180                
      CALL SRSORT (P,TEMP,V,H,S,ISAT,VG,HG,SG )                          H2OSCN    181                
      IF ( S.GT. SLO .AND. S.LT. SHI ) GO TO 45                          H2OSCN    182                
      PX= P -(PI-P) / (SI-S) * (S- XIND )                                H2OSCN    183                
      IF ( JQ.EQ.2) GO TO 62                                             H2OSCN    184                
      IF ( PX.LT.PA ) PX = PA * ( 1. + .05 / I )                         H2OSCN    185                
      GO TO 63                                                           H2OSCN    186                
   62 IF ( PX .GT. PA ) PX = PA * ( 1. - .05 / I )                       H2OSCN    187                
   63 SI = S                                                             H2OSCN    188                
      PI = P                                                             H2OSCN    189                
   67 P  = PX                                                            H2OSCN    190                
      GO TO 199                                                          H2OSCN    191                
   64 IF ( XIND .LT. SG ) GO TO 61                                       H2OSCN    192                
      P= PA - 50.                                                        H2OSCN    193                
      SI = SG                                                            H2OSCN    194                
      JQ = 2                                                             H2OSCN    195                
      GO TO 60                                                           H2OSCN    196                
   61 PRESS= PA                                                          H2OSCN    197                
      ENTR= XIND                                                         H2OSCN    198                
      X = ( XIND - SF ) / ( SG -SF )                                     H2OSCN    199                
      VOL  = VF + X * (VG - VF )                                         H2OSCN    200                
      ENTH = HF + X * (HG - HF )                                         H2OSCN    201                
      GO TO 200                                                          H2OSCN    202                
 200  IF ( JS .GT. 1 ) STOP                                              H2OSCN    203                
      RETURN                                                             H2OSCN    204                
      END                                                                H2OSCN    205                
