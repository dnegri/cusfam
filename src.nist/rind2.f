C
      SUBROUTINE RIND2(ALAMBR, RBAR, TBAR, RINDX)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** RIND2 ***                                    C
C THIS SUBROUTINE COMPUTES THE REFRACTIVE INDEX OF WATER AT A          C
C GIVEN REDUCED TEMPERATURE, REDUCED DENSITY, AND REDUCED WAVELENGTH   C
C ACCORDING TO THE REFIT EQUATION OF SCHIEBENER ET AL                  C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C ALAMBR  R    I   REDUCED WAVELENGTH                                  C
C RBAR    R    I   REDUCED DENSITY                                     C
C TBAR    R    I   REDUCED TEMPERATURE                                 C
C RINDX   R    O   REFRACTIVE INDEX                                    C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 19AUG96 - INITIAL IMPLEMENTATION BY AHH                              C
C 27MAY97 - AHH: CHANGE ERROR/WARNING HANDLING                         C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (ALAMUV=.229202D0, ALAMIR=5.432937D0,
     >           A0=.244257733D0, A1=9.74634476D-3, A2=-3.73234996D-3,
     >           A3=2.68678472D-4, A4=1.58920570D-3, A5=2.45934259D-3,
     >           A6=.900704920D0, A7=-1.66626219D-2)
C
      IF ((ALAMBR .LE. ALAMUV) .OR. (ALAMBR .GE. ALAMIR)) THEN
        RINDX = 0.D0
      ELSE
        ALAM2 = ALAMBR*ALAMBR
        RHS = A0 + A1*RBAR + A2*TBAR + A3*ALAM2*TBAR + A4/ALAM2
     >      + A5/(ALAM2-ALAMUV**2) + A6/(ALAM2-ALAMIR**2) + A7*RBAR**2
        RRHS = RBAR*RHS
        RINDX = DSQRT((2.D0*RRHS + 1.D0) / (1.D0 - RRHS))
      ENDIF
      RETURN
      END
