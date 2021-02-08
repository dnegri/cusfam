      FUNCTION PRLIQ(P,T)                                                PRLIQ       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  34
CPRLI PRANDTL NUMBER OF COMPRESSED LIQUID VS P AND T                     PRLIQ       3                
C                                                                        PRLIQ       4                
C                                                                        PRLIQ       5                
C     FIRST ARGUMENT IS PRESSURE                                         PRLIQ       6                
C     SECOND ARGUMENT IS TEMPERATURE                                     PRLIQ       7                
C     RETURNS WITH PRANDTL NUMBER                                        PRLIQ       8                
C                                                                        PRLIQ       9                
      PRLIQ=VISL(P,T)*1800.*(.08333333*(HPTL(P,T-4.)-HPTL(P,T+4.))       PRLIQ      10                
     1 +.6666666*(HPTL(P,T+2.)-HPTL(P,T-2.)))/CONDL(P,T)                 PRLIQ      11                
      RETURN                                                             PRLIQ      12                
      END                                                                PRLIQ      13                
