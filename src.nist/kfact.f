      SUBROUTINE KFACT(TR, RHOL, RHOV, FLIQ, FVAP, XK, IWANT, PROPR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** KFACT ***                                    C
C THIS ROUTINE COMPUTES THE K-FACTOR AS THE RATIO OF FUGACITY          C
C COEFFICIENTS AT A GIVEN REDUCED TEMPERATURE TR AND LIQUID AND VAPOR  C
C DENSITIES RHOL AND RHOV.                                             C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TR      R    I   REDUCED TEMPERATURE, T/TC                           C
C RHOL    R    I   REDUCED LIQUID DENSITY, RHOL/RHOC                   C
C RHOV    R    I   REDUCED VAPOR DENSITY, RHOV/RHOC                    C
C FLIQ    R    O   LIQUID FUGACITY COEFFICIENT                         C
C FVAP    R    O   VAPOR FUGACITY COEFFICIENT                          C
C XK      R    O   K-FACTOR = FLIQ/FVAP                                C
C IWANT   IA   -   PROPERTY REQUEST VECTOR FOR USE IN PROPS2 ROUTINE   C
C PROPR   RA   O   VECTOR OF REDUCED PROPERTIES AS REQUESTED BY IWANT  C
C                  TO BE RETURNED BY THE PROP2 ROUTINE                 C
C                  RELEVANT PROPERTIES HERE ARE:                       C
C                 13: FUGACITY COEFFICIENT                             C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 20JUL95 - INITIAL CREATION BY AHH                                    C
C 05SEP95 - AHH: CHANGE TO TR IN PROPS2 ARGUMENT LIST                  C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 11DEC95 - AHH: RENUMBERING OF PROPERTY ARRAY                         C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 08JAN97 - AHH: CHANGE IN ARGUMENT LIST OF PROPS2                     C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWANT(NPROP)
C
C SET IWANT VECTOR TO RETURN FUGACITY COEFFICIENT
C
      CALL IVZERO(IWANT, NPROP)
      IWANT(13) = 1
C
C CALL PROP2 TO RETURN VAP AND LIQ FUGACITY COEFFICIENTS
C
      CALL PROPS2(IWANT, 0, TR, RHOL, PROPR)
      FLIQ = PROPR(13)
      CALL PROPS2(IWANT, 0, TR, RHOV, PROPR)
      FVAP = PROPR(13)
C
      XK = FLIQ / FVAP
C
      RETURN
      END
