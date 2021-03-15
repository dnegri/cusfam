!**********************************************************************!
!                                                                      !
! KEPCO Nuclear Fuel CONFIDENTIAL                                      !
! __________________                                                   !
!                                                                      !
!  [2007] - [2013] KEPCO Nuclear Fuel Incorporated                     !
!  All Rights Reserved.                                                !
!                                                                      !
! NOTICE:  All information contained herein is, and remains the        ! 
!          property of KEPCO Nuclear Fuel Incorporated and its         !
!          suppliers, if any. The intellectual and technical           !
!          concepts contained herein are proprietary to KEPCO          !
!          Nuclear Fuel Incorporated and protected by trade            !
!          secret or copyright law.                                    !
!          Dissemination of this information or reproduction of        !
!          this material is strictly forbidden unless prior            !
!          written permission is obtained from KEPCO Nuclear           !
!          Fuel Incorporated.                                          !
!**********************************************************************!





      SUBROUTINE QUAD1( N   , X   , XVAL, KLO , AF)
!
!***********************************************************************
!*                                                                     *
!*    ROUTINE NAME      : QUAD1                                        *
!*    ROUTINE TYPE      : SUBROUTINE                                   *
!*                                                                     *
!***********************************************************************
!*                                                                     *
!*    AUTHOR(S)         : C.H.LEE                                      *
!*    DEPARTMENT        : NGR/ADONIS                                   *
!*                                                                     *
!***********************************************************************
!*                                                                     *
!*    DESCRIPTION       : QUADRATIC INTERPOLATION                      *
!*                                                                     *
!***********************************************************************
!*                                                                     *
!*    DESCRIPTION OF PARAMETERS                                        *
!*                                                                     *
!*    NAME     TYPE       DESCRIPTION                                  *
!*                                                                     *
!*    KLO      INTEGER    ARRAY NUMBER CORRESPONDING TO af(1)            *
!*    af(1)      REAL       1ST COEFFICIENT FOR INTERPOLATION            *
!*    af(2)      REAL       2ND COEFFICIENT FOR INTERPOLATION            *
!*    af(3)      REAL       3RD COEFFICIENT FOR INTERPOLATION            *
!*                                                                     *
!***********************************************************************
!
      REAL(4) X(N)
      real(XS_PREC) af(3)
      real(XS_PREC) XVAL
!
! --- X(N) IN ASCENDING ORDER
!
      if (n .eq. 1) then
         KLO = N
         af(1) = 1.0
         af(2) = 0.0
         af(3) = 0.0
         RETURN
      ENDIF

!
! --- EXTRAPOLATION
!
      IF (XVAL .GT. X(N)) THEN
         KLO=N-2
         af(1)=0.
         af(3)=(XVAL-X(N-1))/(X(N)-X(N-1))
         af(2)=1.-af(3)
         RETURN
      ELSE IF (XVAL .LT. X(1)) THEN
         KLO=1
         af(2)=(XVAL-X(  1))/(X(2)-X(  1))
         af(1)=1.-af(2)
         af(3)=0.
         RETURN
      ENDIF
!
! --- INTERPOLATION
!
      IF (XVAL .LT. X(2)) THEN
         KLO=1
         KHI=2
         GOTO 280
      ELSE IF (XVAL .GT. X(N-2)) THEN
         KLO=N-2
         KHI=N-1
         GOTO 280
      ENDIF
!
      KLO=2
      KHI=N-2
!
 1000 IF(KHI-KLO.GT.1) THEN
         K=(KHI+KLO)/2
         IF(X(K).GT.XVAL) THEN
            KHI=K
         ELSE
            KLO=K
         ENDIF
         GOTO 1000
      ENDIF
!
  280 CONTINUE
      H=X(KHI)-X(KLO)
      IF(H.EQ.0) THEN
         WRITE(*,*) 'XVAL = ', XVAL
         WRITE(*,*) 'X(N) = ', X, N
         stop
      ENDIF
!
      AB1    = XVAL
      A1     = X(KLO  )
      A2     = X(KHI  )
      A3     = X(KHI+1)
!
!     WRITE(*,*) A1,A2,A3,AB1
!
      A12    = A1 - A2
      A21    = -A12
      A23    = A2 - A3
      A32    = -A23
      A31    = A3 - A1
      A13    = -A31
      XA1    = AB1-A1
      XA2    = AB1-A2
      XA3    = AB1-A3
      af(1)    = XA2*XA3/(A12*A13)
      af(2)    = XA1*XA3/(A21*A23)
      af(3)    = XA1*XA2/(A31*A32)
!
      RETURN
      END
