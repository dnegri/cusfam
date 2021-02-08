      SUBROUTINE DFIND(DOUT, P, D, TR, DPD, IWANT, PROPR, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** DFIND ***                                    C
C THIS ROUTINE FINDS THE REDUCED DENSITY CORRESPONDING TO A GIVEN      C
C REDUCED TEMPERATURE TR AND REDUCED (P/PC) PRESSURE P.  AN            C
C INITIAL GUESS (D) IS REQUIRED, AND THE VALUE OF DPDD IS ALSO         C
C RETURNED.  ROUTINE ADAPTED DIRECTLY FROM OLD (1984) STEAM PROGRAM.   C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C DOUT    R    O   REDUCED DENSITY RHO/RHOC AT TAU AND P               C
C P       R    I   REDUCED (P/PC) PRESSURE                             C
C D       R    I   INITIAL GUESS FOR REDUCED DENSITY, RHO/RHOC         C
C TR      R    I   REDUCED TEMPERATURE, T/TC                           C
C DPD     R    O   FIRST DERIVATIVE OF P WITH RESPECT TO D AT DOUT     C
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
C 12JUL95 - ADAPTED FROM OLD STEAM PROGRAM BY AHH                      C
C 20JUL95 - AHH: IWANT AND PROPR DIMENSIONED TO 19                     C
C 21AUG95 - AHH: CHANGE TO CALL DFIND1 FOR SUPERCRITICAL TEMPERATURES  C
C 28AUG95 - AHH: CHANGE TO CALL DFIND2 FOR SUBCRITICAL TEMPERATURES    C
C 30AUG95 - AHH: ADD IERR FLAG TO RETURN STATUS                        C
C 06SEP95 - AHH: USE TR INSTEAD OF TAU IN ARGUMENT LIST                C
C 28SEP95 - AHH: PARAMETERIZE NUMBER OF PROPERTIES                     C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
      DIMENSION PROPR(NPROP)
      DIMENSION IWANT(NPROP)
      IF (TR .GE. 1.D0) THEN
        CALL DFIND1(DOUT,P,D,TR,DPD,IWANT,PROPR,IERR)
      ELSE
        CALL DFIND2(DOUT,P,D,TR,DPD,IWANT,PROPR,IERR)
      ENDIF
      RETURN
      END
