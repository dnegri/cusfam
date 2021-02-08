      SUBROUTINE STER(ANAME, I, A, B)                                     STER        2                
      IMPLICIT REAL*8 (A-H,O-Z)                                          STEAMWV1M0(LIB)-NOV. 1,90  51
C  STER  ROUTINE FOR PRINTING ERROR MESSAGES ASSOCIATED                  STER        3                
C        WITH THE STEAM TABLES                                           STER        4                
C                                                                        STER        5                
C                                                                        STER        6                
C     NAME IS THE BCD NAME OF THE CALLING ROUTINE.                       STER        7                
C     I POSITIVE PRINTS -OUT OF RANGE-                                   STER        8                
C     I NEGATIVE PRINTS -NON CONVERGENT-                                 STER        9                
C     IF I IS ONE DIGIT ITS CORRESPONDING ARGUMENT A IS PRINTED.         STER       10                
C     IF I IS TWO DIGITS ITS TWO CORR. ARGUMENTS A AND B ARE PRINTED.    STER       11                
c      REAL*8 MA(5)                                                       STEAMWV1M0(LIB)-NOV. 1,90  52
c      REAL*8 MP                                                          STEAMWV1M0(LIB)-NOV. 1,90  53
c      REAL*8 NAME                                                        STEAMWV1M0(LIB)-NOV. 1,90  54
c      DIMENSION M(10)                                                    STER       15                
      character ANAME*6,AMA(5)*4,AM(10)*4,AMP*4
      DATA AM/4H OUT,4H OF ,4HRANG,4HE IN,4H    ,4HNON ,4HCONV,4HERGE,    STER       16                
     1 4HNT I, 4HN    /                                                  STER       17                
      DATA AMA/6HPRESS=,6HTEMP =,6HENTH =,5HVOL =,6HENTR =/               STER       18
C                                                                        STEAMV1M2 - 3/96
      SAVE AM,AMA                                                          STEAMV1M2 - 3/96
C                                                                        STEAMV1M2 - 3/96
      IM = 1                                                             STER       19                
      IF(I.LT.0) IM=6                                                    STER       20                
      I2 = IABS(I)                                                       STER       21                
      I1 = I2/10                                                         STER       22                
      I2 = I2 - I1 * 10                                                  STER       23                
      AMP = AMA(I2)                                                        STER       24                
      IF(I1.GT. 0) AMP=AMA(I1)                                             STER       25                
      WRITE(6,8) AM(IM),AM(IM+1),AM(IM+2),AM(IM+3),AM(IM+4),ANAME,AMP,A    STER       26                
      IF(I1 .EQ. 0) GO TO 4                                              STER       27                
      WRITE(6,9) AMA(I2),B                                               STER       28                
    4 STOP                                                               STER       29                
    8 FORMAT(1X, 4A4,A1,A6,2X,A6, E13.6)                                 STER       30                
    9 FORMAT(27X, A6, E13.6)                                             STER       31                
      END                                                                STER       32                
