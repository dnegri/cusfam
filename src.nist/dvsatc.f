C
      FUNCTION DVSATC(TR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** DVSATC ***                                   C
C THIS ROUTINE ESTIMATES THE REDUCED SAT. VAPOR  DENSITY AT A GIVEN    C
C REDUCED TEMPERATURE T/TC USING A POWER-LAW FORMULA THAT INTERPOLATES C
C FROM 647.095 K TO THE CRITICAL POINT                                 C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TR      R    I   REDUCED TEMPERATURE T/TC                            C
C DVSATC  R    O   REDUCED SATURATED VAPOR  DENSITY RHO/RHOC           C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 23MAY96 - INITIAL CREATION BY AHH                                    C
C 10NOV99 - AHH: READJUSTMENT FOR SMOOTHER JOINING                     C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (A2 = 12.9987D0, BETA = 0.5D0)
      TAU = 1.D0 - TR
      DVSATC = 1.D0 - A2*TAU**BETA
      RETURN
      END
