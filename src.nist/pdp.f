C
      SUBROUTINE PDP(DEL, TR, PR, DPRDD, IWANT, PROPR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** PDP ***                                      C
C THIS ROUTINE COMPUTES THE REDUCED PRESSURE P/PC AND ITS FIRST        C
C DERIVATIVE WITH RESPECT TO REDUCED DENSITY AT A GIVEN REDUCED        C
C TEMPERATURE TR AND REDUCED DENSITY DEL.                              C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C DEL     R    I   REDUCED DENSITY, RHO/RHOC                           C
C TR      R    I   REDUCED TEMPERATURE, T/TC                           C
C PR      R    O   REDUCED PRESSURE, P/PC                              C
C DPRDD   R    O   FIRST DERIVATIVE OF PR WITH RESPECT TO DEL          C
C IWANT   IA   -   PROPERTY REQUEST VECTOR FOR USE IN PROPS2 ROUTINE   C
C PROPR   RA   O   VECTOR OF REDUCED PROPERTIES AS REQUESTED BY IWANT  C
C                  TO BE RETURNED BY THE PROP2 ROUTINE                 C
C                  RELEVANT PROPERTIES HERE ARE:                       C
C                  2: REDUCED PRESSURE P/(RHO*R*T)                     C
C                 16: REDUCED DP/DRHO DPDR/RT                          C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 12JUL95 - INITIAL CREATION BY AHH                                    C
C 20JUL95 - AHH: PROPR AND IWANT NOW DIMENSIONED TO 19                 C
C 05SEP95 - AHH: CHANGE TO TR IN PROPS2 ARGUMENT LIST                  C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 11DEC95 - AHH: RENUMBERING OF PROPERTY VECTOR                        C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 08JAN97 - AHH: CHANGE IN ARGUMENT LIST OF PROPS2                     C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
C
C NEED CONSTANTS FOR WATER 
C
      INCLUDE 'wconst.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWANT(NPROP)
C
C SET IWANT VECTOR TO RETURN P AND DPDD
C
      CALL IVZERO(IWANT, NPROP)
      IWANT(2) = 1
      IWANT(16) = 1
C
C CALL PROPS2 TO RETURN DIMENSIONLESS PROPERTIES
C
      CALL PROPS2(IWANT, 0, TR, DEL, PROPR)
C
C THIS RETURNED PRESSURES REDUCED BY RHO*RT, SO MUST CONVERT TO PC
C AS REDUCING BASIS
C
      RHO = DEL*RHOCW
      RTW = RW*TR*TCW
      PR = PROPR(2)*RHO*RTW*1.D-3/PCW
      DPRDD = PROPR(16)*RTW*1.D-3/PCW*RHOCW
      RETURN
      END
