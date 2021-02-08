C
      SUBROUTINE SURF(TK, SURFT, IERR)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** SURF ***                                     C
C THIS ROUTINE COMPUTES THE VAPOR/LIQUID SURFACE TENSION               C
C OF WATER AT A GIVEN TEMPERATURE AND AND SATURATION CONDITIONS        C
C ACCORDING TO THE SEPTEMBER 1994 IAPWS RELEASE ON THE SURFACE TENSION C
C OF ORDINARY WATER SUBSTANCE                                          C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    I   TEMPERATURE IN KELVINS                              C
C SURFT   R    O   SURFACE TENSION IN N/M                              C
C IERR    I    O   RETURN CODE:                                        C
C                  0 = SUCCESS                                         C
C                  1 = TEMPERATURE BELOW LOWER LIMIT FOR SATURATION    C
C                      CALCULATION, ZERO RETURNED                      C
C                  2 = TEMPERATURE ABOVE CRITICAL POINT, ZERO RETURNED C
C                  3 = 2-PHASE REGION EXTRAPOLATED BELOW TRIPLE POINT  C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 19OCT95 - INITIAL IMPLEMENTATION BY AHH                              C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 27MAY97 - AHH: IMPLEMENT IERR=3                                      C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'wconst.cmn'
      INCLUDE 'wlimit.cmn'
C
      IF (TK .LT. TSMIN) THEN
        IERR = 1
        SURFT = 0.D0
      ELSE IF (TK .GT. TCW) THEN
        IERR = 2
        SURFT = 0.D0
      ELSE
        IF (TK .LT. TTRIPW) THEN
          IERR = 3
        ELSE
          IERR = 0
        ENDIF
        TAU = 1.D0 - TK/TCW
        SURFT = 0.2358D0 * (TAU**1.256D0) * (1.D0 - 0.625D0*TAU)
      ENDIF
      RETURN
      END
