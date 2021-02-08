C
      SUBROUTINE IVZERO(IVEC, ILEN)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                     *** IVZERO ***                                   C
C THIS ROUTINE ZEROS OUT AN INTEGER VECTOR OF LENGTH ILEN              C
C                                                                      C
C ARGUMENT LIST:                                                       C
C  NAME  TYPE I/O? EXPLANATION                                         C
C ------ ---- ---- --------------------------------------------------  C
C IVEC    IA  I/O  INTEGER VECTOR TO BE ZEROED                         C
C ILEN    I    I   LENGTH OF IVEC                                      C
C                                                                      C
C MAINTENANCE:                                                         C
C                                                                      C
C 12JUL95 - INITIAL CREATION BY AHH                                    C
C                                                                      C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      DIMENSION IVEC(ILEN)
      DO 100 I=1,ILEN
  100 IVEC(I) = 0
      RETURN
      END
