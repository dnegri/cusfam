C
      SUBROUTINE BNDCHK(TK, PMPA, RHO, MODE, ISCHK, ISFLG, ICCHK, ICFLG,
     >                  IPCHK, IPFLG, IWORK, PROPSI, RWORK)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** BNDCHK ***                                   C
C THIS ROUTINE CHECKS TO SEE IF THE REQUESTED POINT IS AN EXTRAPOLATIONC
C OR OUT OF RANGE FOR A VARIETY OF CRITERIA                            C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C TK      R    I   TEMPERATURE IN KELVINS                              C
C PMPA    R    I   PRESSURE IN MPA                                     C
C RHO     R    I   DENSITY IN KG/M3                                    C
C MODE    I    I   TELLS WHAT INFORMATION IS AVAILABLE ON ENTRY        C
C                  0: T, P, RHO ALL KNOWN                              C
C                  1: T AND P KNOWN, RHO UNKNOWN                       C
C                  2: T AND RHO KNOWN, P UNKNOWN                       C
C *** NOTE:  FOLLOWING ARE FLAGS FOR WHETHER TO MAKE A CHECK (1=CHECK, C
C            0=DON'T) AND FLAGS RETURNED FROM THE CHECKS (0=OK OR NOT  C
C            CHECKED, 1=EXTRAPOLATED BEYOND RECOMMENDED REGION, 2=     C
C            UNACCEPTABLY OUTSIDE OF REGION                            C
C ISCHK   I    I   WHETHER TO CHECK IF EQUILIBRIUM PHASE IS SOLID      C
C ISFLG   I    O   RESULT OF SOLID CHECK                               C
C ICCHK   I    I   WHETHER TO CHECK IF TOO NEAR CRITICAL POINT         C
C ICFLG   I    O   RESULT OF NEAR-CRITICAL CHECK                       C
C IPCHK   IA   I   WHETHER TO CHECK BOUNDARIES FOR VARIOUS PROPERTIES. C
C                  ARRAY ELEMENTS ARE: 1: THERMODYNAMIC PROPERTIES     C
C                                      2: VISCOSITY                    C
C                                      3: THERMAL CONDUCTIVITY         C
C                                      4: DIELECTRIC CONSTANT          C
C                                      5: REFRACTIVE INDEX             C
C IPFLG   IA   O   RESULTS OF PROPERTY BOUNDARY CHECKS                 C
C IWORK   IA   -   INTEGER WORK ARRAY, USED IF P GETS CALCULATED       C
C PROPSI  RA   O   VECTOR OF PROPERTIES IN SI UNITS AS REQUESTED BY    C
C                  IWANT.  NUMBERING OF PROPERTIES:                    C
C                  2: PRESSURE IN MPA                                  C
C RWORK   RA   -   REAL WORK ARRAY, USED IF P GETS CALCULATED          C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 08NOV95 - INITIAL CREATION BY AHH                                    C
C 21MAY96 - AHH: PUT IN NEAR-CRITICAL CHECK, MODIFY OTHER BOUNDARIES   C
C           IN ACCORDANCE WITH NEWLY RECEIVED RELEASE LANGUAGE         C
C 30AUG96 - AHH: LOWERCASE INCLUDES FOR OTHER PLATFORMS                C
C 08JAN97 - AHH: CHANGE IN ARGUMENT LIST OF PROPS1                     C
C 25MAR97 - AHH: ADD CHECK FOR REFRACTIVE INDEX BOUNDARY               C
C 10SEP03 - AHH: CORRECT LOW-T BOUNDARY FOR THERMO PROPS (MELTING LINE)C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INCLUDE 'nprop.cmn'
C
C NEED CONSTANTS FOR WATER 
C
      INCLUDE 'wconst.cmn'
      INCLUDE 'wlimit.cmn'
      PARAMETER (RCMNS=320.39D0, RCPLS=323.61D0, TCMNS=647.076D0,
     >           TCPLS=652.096D0)
      DIMENSION PROPSI(NPROP), RWORK(NPROP)
      DIMENSION IWORK(NPROP), IPCHK(5), IPFLG(5)
      ISFLG = 0
      ICFLG = 0
      DO 10 I=1,5
        IPFLG(I) = 0
   10 CONTINUE
      TC = TK - 273.15D0
C
C FIRST USE MODE TO DETERMINE WHAT YOU HAVE AND WHAT YOU NEED
C
      IF (MODE .EQ. 0) THEN
        IHAVEP = 1
        IHAVER = 1
      ELSE IF (MODE .EQ. 1) THEN
        IHAVEP = 1
        IHAVER = 0
      ELSE
        IHAVEP = 0
        IHAVER = 1
      ENDIF
      IF ((IHAVEP.EQ.0) .AND. ((ISCHK.EQ.1) .OR. (IPCHK(1).EQ.1) .OR.
     >    (IPCHK(2).EQ.1) .OR. (IPCHK(3).EQ.1) .OR. (IPCHK(4).EQ.1) .OR.
     >    (IPCHK(5).EQ.1))) THEN
        INEEDP = 1
      ELSE
        INEEDP = 0
      ENDIF
      IF ((IHAVER.EQ.0) .AND. ((ICCHK.EQ.1) .OR. (IPCHK(5).EQ.1))) THEN
        INEEDR = 1
      ELSE
        INEEDR = 0
      ENDIF
C
C CHECK FOR "BAD" VALUES OF T AND RHO
C
      IF (TK .LT. 0.D0) THEN
        IF (ISCHK.EQ.1) ISFLG = 2
        DO 20 I=1,5
          IF (IPCHK(I) .EQ. 1) IPFLG(I) = 2
   20   CONTINUE
        RETURN
      ENDIF
      IF (IHAVER .EQ. 1) THEN
        IF (RHO .LE. 0.D0) THEN
          DO 30 I=1,5
            IF (IPCHK(I) .EQ. 1) IPFLG(I) = 2
   30     CONTINUE
          RETURN
        ENDIF
      ENDIF
C
C COMPUTE P OR RHO IF NEEDED
C
            
      IF (INEEDP .EQ. 1) THEN
        CALL IVZERO(IWORK, NPROP)
        IWORK(2) = 1
        CALL PROPS1(IWORK, 0, TK, RHO, PROPSI, RWORK)
        PMPA = PROPSI(2)
      ENDIF
      IF (INEEDR .EQ. 1) THEN
        CALL DENS0(RHO, PMPA, TK, DPD, IWORK, RWORK, IERRD)
      ENDIF
C
C CHECK FOR SOLID BOUNDARIES
C
      IF (ISCHK .EQ. 1) THEN
C
C CHECK FOR WHETHER EQUILIBRIUM STATE WOULD BE A SOLID
C DEPENDING ON PRESSURE, CHECK VERSUS SUBLIMATION OR FREEZING
C STILL NEED TO DECIDE HOW MUCH SUBCOOLING IS "RELIABLE"
C
        IF (PMPA .LE. PTRIPW) THEN
          CALL TSUB(TSOLID, PMPA, IERR)
          IF (IERR .NE. 1) THEN
            IF (TK .LT. (TSOLID-35.D0)) THEN
              ISFLG = 2
            ELSE IF (TK .LT. TSOLID) THEN
              ISFLG = 1
            ENDIF
          ENDIF
        ELSE
          CALL TMELT(TSOLID, PMPA, IFORM, IERR)
          IF (IERR .NE. 2) THEN
            IF (TK .LT. (TSOLID - 35.D0)) THEN
              ISFLG = 2
            ELSE IF (TK .LT. TSOLID) THEN
              ISFLG = 1
            ENDIF
          ENDIF
        ENDIF        
      ENDIF
C
C CHECK FOR NEAR-CRITICAL REGION
C
      IF ((ICCHK .EQ. 1) .AND. ((IERRD.EQ.0) .OR. (ABS(IERRD).EQ.3)))
     >    THEN
        IF ((TK.GT.TCMNS) .AND. (TK.LT.TCPLS) .AND. (RHO.GT.RCMNS)
     >      .AND. (RHO.LT.RCPLS)) ICFLG = 1
      ENDIF
C
C CHECKS FOR INDIVIDUAL PROPERTIES OR SETS OF PROPERTIES
C
C CHECK FOR BOUNDS OF WAGNER/PRUSS EQUATION
C 
      IF (IPCHK(1) .EQ. 1) THEN
        IT = 0
        IP = 0
        IF (PMPA .LE. PTRIPW) THEN
          CALL TSUB(TSOLID, PMPA, IERR)
        ELSE
          CALL TMELT(TSOLID, PMPA, IFORM, IERR)
        ENDIF
        IF ((TK .GT. TUPPER) .OR. (TK .LT. TLOWER)) THEN
          IT = 2
        ELSE IF ((TK .GT. 1273.15D0) .OR. (TK .LT. TSOLID)) THEN
          IT = 1
        ENDIF
        IF ((PMPA .GT. PUPPER) .OR. (PMPA .LT. PLOWER)) THEN
          IP = 2
        ELSE IF ((PMPA .GT. 1.D3) .OR. (PMPA .LE. 0.D0)) THEN
          IP = 1
        ENDIF
        IPFLG(1) = MAX(IP, IT)
      ENDIF
C
C CHECK FOR BOUNDS OF VISCOSITY CORRELATION
C
      IF (IPCHK(2) .EQ. 1) THEN
        IT = 0
        IP = 0
        IF ((TK .GT. TUPPER) .OR. (TK .LT. TLOWER)) THEN
          IT = 2
        ELSE IF ((TC .GT. 900.D0) .OR. 
     >           ((TC .GT. 600.D0) .AND. (PMPA .GT. 300.D0)) .OR.
     >           ((TC .GT. 150.D0) .AND. (PMPA .GT. 350.D0)) .OR.
     >           (PMPA .GT. 500.D0) .OR. (TC .LT. 0.D0)) THEN
          IT = 1
        ENDIF
        IF ((PMPA .GT. PUPPER) .OR. (PMPA .LT. PLOWER)) THEN
          IP = 2
        ELSE IF (PMPA .LE. 0.D0) THEN
          IP = 1
        ENDIF
        IPFLG(2) = MAX(IP, IT)
      ENDIF
C
C CHECK FOR BOUNDS OF THERMAL CONDUCTIVITY CORRELATION
C
      IF (IPCHK(3) .EQ. 1) THEN
        IT = 0
        IP = 0
        IF ((TK .GT. TUPPER) .OR. (TK .LT. TLOWER)) THEN
          IT = 2
        ELSE IF ((TC .GT. 800.D0) .OR. 
     >           ((TC .GT. 400.D0) .AND. (PMPA .GT. 100.D0)) .OR.
     >           ((TC .GT. 250.D0) .AND. (PMPA .GT. 150.D0)) .OR.
     >           ((TC .GT. 125.D0) .AND. (PMPA .GT. 200.D0)) .OR.
     >           (PMPA .GT. 400.D0) .OR. (TC .LT. 0.D0)) THEN
          IT = 1
        ENDIF
        IF ((PMPA .GT. PUPPER) .OR. (PMPA .LT. PLOWER)) THEN
          IP = 2
        ELSE IF (PMPA .LE. 0.D0) THEN
          IP = 1
        ENDIF
        IPFLG(3) = MAX(IP, IT)
      ENDIF
C
C CHECK FOR BOUNDS OF DIELECTRIC CONSTANT CORRELATION
C
      IF (IPCHK(4) .EQ. 1) THEN
        IT = 0
        IP = 0
        IF ((TK .GT. TUPPER) .OR. (TK .LT. TLOWER)) THEN
          IT = 2
        ELSE IF ((TK .GT. 873.D0) .OR. (TK .LT. 238.D0)) THEN
          IT = 1
        ENDIF
        IF ((PMPA .GT. PUPPER) .OR. (PMPA .LT. PLOWER)) THEN
          IP = 2
        ELSE IF ((PMPA .GT. 1.D3) .OR. (PMPA .LE. 0.D0)) THEN
          IP = 1
        ENDIF
        IPFLG(4) = MAX(IP, IT)
      ENDIF
C
C CHECK FOR BOUNDS OF REFRACTIVE INDEX CORRELATION
C
      IF (IPCHK(5) .EQ. 1) THEN
        IT = 0
        IP = 0
        IF ((TK .GT. TUPPER) .OR. (TK .LT. TLOWER)) THEN
          IT = 2
        ELSE IF ((TK .GT. 773.15D0) .OR. (TK .LT. 261.15D0)) THEN
          IT = 1
        ENDIF
        IF ((PMPA .GT. PUPPER) .OR. (PMPA .LT. PLOWER)) THEN
          IP = 2
        ELSE IF (RHO .GT. 1060.D0) THEN
          IP = 1
        ENDIF
        IPFLG(5) = MAX(IP, IT)
      ENDIF
C
      RETURN
      END
