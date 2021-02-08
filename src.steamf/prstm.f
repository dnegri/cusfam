      FUNCTION PRSTM(P,T)                                                PRSTM       2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  35
CPRST PRANDTL NUMBER OF SUPERHEATED STEAM VS P AND T                     PRSTM       3                
C                                                                        PRSTM       4                
C                                                                        PRSTM       5                
C     FIRST ARGUMENT IS PRESSURE                                         PRSTM       6                
C     SECOND ARGUMENT IS TEMPERATURE                                     PRSTM       7                
C     RETURNS WITH PRANDTL NUMBER                                        PRSTM       8                
C                                                                        PRSTM       9                
      PRSTM=VISV(P,T)*1800.*(.08333333*(HPTD(P,T-4.)-HPTD(P,T+4.))       PRSTM      10                
     1 +.6666666*(HPTD(P,T+2.)-HPTD(P,T-2.)))/CONDV(P,T)                 PRSTM      11                
      RETURN                                                             PRSTM      12                
      END                                                                PRSTM      13                
