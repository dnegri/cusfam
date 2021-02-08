C
      SUBROUTINE DFIND0(DOUT, PR, TR, DPD, IWANT, PROPR, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** DFIND0 ***                                   C
C THIS ROUTINE FINDS THE REDUCED DENSITY CORRESPONDING TO A GIVEN      C
C REDUCED TEMPERATURE TR AND REDUCED (P/PC) PRESSURE PR.  IN           C
C CONTRAST TO DFIND, NO INITIAL GUESS IS REQUIRED (ONE IS GENERATED    C
C INTERNALLY).  THE VALUE OF DPD IS ALSO RETURNED.                     C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C DOUT    R    O   REDUCED DENSITY RHO/RHOC AT TAU AND PR              C
C PR      R    I   REDUCED (P/PC) PRESSURE                             C
C TR      R    I   REDUCED TEMPERATURE, T/TC                           C
C DPD     R    O   FIRST DERIVATIVE OF PR WITH RESPECT TO D AT DOUT    C
C IWANT   IA   -   PROPERTY REQUEST VECTOR FOR USE IN PROPS2 ROUTINE   C
C PROPR   RA   O   VECTOR OF REDUCED PROPERTIES AS REQUESTED BY IWANT  C
C                  TO BE RETURNED BY THE PROP2 ROUTINE                 C
C                  RELEVANT PROPERTIES HERE ARE:                       C
C                  2: REDUCED PRESSURE P/(RHO*R*T)                     C
C                 12: REDUCED DP/DRHO DPDR/RT                          C
C IERR    I    O   RETURN STATUS FLAG.  NEGATIVES ARE FROM DFIND1,     C
C                  POSITIVES FROM DFIND2.  MEANINGS:                   C
C                  0: CONVERGED                                        C
C                 -2: UNABLE TO BOUND ROOT (DFIND1)                    C
C                 -3: UNABLE TO CONVERGE BOUNDED ROOT (DFIND1)         C
C                  1: NO ROOT FOR REQUESTED PHASE.  RETURNS D WHERE    C
C                     DPD = 0 (DFIND2)                                 C
C                  2: UNABLE TO BOUND ROOT (DFIND2)                    C
C                  3: UNABLE TO CONVERGE BOUNDED ROOT (DFIND2)         C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 30AUG95 - INITIAL CREATION FROM DFIND BY AHH                         C
C 06SEP95 - AHH: USE TR INSTEAD OF TAU IN ARGUMENT LIST                C
C 13SEP95 - AHH: FOR SUBCRITICAL, CALL PCOEX TO GET EXACT V-L BORDER   C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 02OCT95 - AHH: IF T TOO LOW FOR PCOEX, USE AUXILIARY EQN. INSTEAD    C
C 03OCT95 - AHH: SPEED - GET INITIAL GUESSES FROM AUXILIARY EQUATIONS  C
C                UNLESS WITHIN 5% OF COEXISTENCE.  ALSO IMPROVE INIT.  C
C                GUESS FOR DENSE SUBCRITICAL VAPORS                    C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 04JUN97 - AHH: ALLOW EXTRAPOLATED LIQUIDS BELOW TRIPLE POINT         C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      INCLUDE 'wconst.cmn'
      INCLUDE 'wlimit.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWANT(NPROP)
      PBOTR = PSMIN / PCW
C 
C IF TR >= 1, USE IDEAL GAS FOR INITIAL GUESS.  ALSO USE IDEAL GAS IF
C IN SUBCOOLED VAPOR REGION.  IF IN COMPRESSED LIQUID REGION, USE
C SATURATED LIQUID DENSITY AS INITIAL GUESS
C IF P BELOW MINIMUM FOR COEXISTENCE CALC, USE VAPOR INITIAL GUESS
C
      IF (TR .LT. 1.D0) THEN
        CALL PVAP(TR, P0AUX, DP0)
        IF (DABS(1.D0-P0AUX/PR) .LT. 0.05D0) THEN
          CALL PCOEX(TR,P0,RHOL,RHOV,IWANT,PROPR,IERR)
          IF (IERR .EQ. 1) THEN
            P0 = P0AUX
            RHOL = DLSAT(TR)
            IF (PR .GT. 0.1) RHOV = DVSAT(TR)
          ENDIF
        ELSE
          P0 = P0AUX
          RHOL = DLSAT(TR)
          IF (PR .GT. 0.1) RHOV = DVSAT(TR)
        ENDIF
        IF ((PR .LT. P0) .OR. (PR .LT. PBOTR)) THEN
          RGUESS = PR*PCW*1.D3 / (RW*TR*TCW)
          RGUESS = RGUESS / RHOCW
          IF (PR .GT. 0.1) RGUESS = RGUESS + (PR/P0)*(RHOV-RGUESS)
        ELSE
          RGUESS = RHOL
        ENDIF
      ELSE
        RGUESS = PR*PCW*1.D3 / (RW*TR*TCW)
        RGUESS = RGUESS / RHOCW
      ENDIF
      CALL DFIND(DOUT,PR,RGUESS,TR,DPD,IWANT,PROPR,IERR)
      RETURN
      END
