      FUNCTION HSS(P,T,S,V)                                              HSS         2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  30
CHSS  SUPERHEATED STEAM H,S, AND V AS A FUNCTION OF P AND T              HSS         3                
C                                                                        HSS         4                
C                                                                        HSS         5                
      V=VPTD(P,T)                                                        HSS         6                
      S=SDRY(P,T)                                                        HSS         7                
      HSS=HDRY(P,T)                                                      HSS         8                
      RETURN                                                             HSS         9                
      END                                                                HSS        10                
