      function steamf(Pref,I1,I2,V1)                                            ! Pref  : Pressure                              [psia]
                                                                                ! I1    : given property option (1 or 2)
      implicit real*8 (a-h,o-z)                                                 ! I2    : wanted property option (1 ~ 7)
                                                                                !         = 8, saturated liquid enthalpy at given pressure
                                                                                !         = 9, saturated vapor enthalpy at given pressure
                                                                                !         = 10, saturated pressure at given temperature
                                                                                !         = 11, surface tension at given temerature
                                                                                ! V1    : given property
      include "block.data"

      ERR = 1.e-10
      IR = 0

      if(Pref.gt.PCA) stop ' P > Critical pressure in steamf'
      if(I1.le.0 .or. I1.gt.3) stop 'I1 must be 1,2 or 3 in steamf'
      if(I2.le.0 .or. I2.gt.11) stop 'I2 must be 1~11 in steamf'
      if(I1.eq.I2) stop 'I1 = I2 in steamf'      

      call SATUR(Pref,Tsat,SVf,Hf,Sf,SVg,Hg,Sg,1)

      if(I2 .eq. 8) then
        steamf = Hf
        return
      endif
      if(I2 .eq. 9) then
        steamf = Hg
        return
      endif
      if(I2 .eq. 10) then
         steamf = PSL(V1)
         return
      endif

      if(I2 .eq. 11) then
         steamf = sigma(V1)
         return
      endif

      go to (1,2,4) I1                                                 
                                                                                 
    1 T = V1                                                                    ! 1, temperature(given property)                [oF]
      call SRSORT(Pref,T,SV,H,S,ISAT,SVg,Hg,Sg )
      go to 10
                
    2 H = V1    
      if(H.le.Hf .or. H.ge.Hg) then
        T = Tsat
        if(H.lt.Hf) then
          T = Tsat-0.00001
        elseif(H.eq.Hf) then
          SV = SVf
          S  = Sf
          go to 10
        elseif(H.eq.Hg) then
          SV = SVg
          S  = Sg
          go to 10
        endif
    3   IR =IR+1
        call SRSORT(Pref,T,SV,Hnew,S,ISAT,SVg,Hg,Sg )
        del = (Hnew-H)/H
        if(abs(del).le.ERR) go to 10
        call bisection(IR,T,Hnew,H,25.d0,del)
        go to 3
      else      
        T = Tsat  
        go to 10
      endif
    4 SV = V1
      if(SV.le.SVf .or. SV.ge.SVg) then
        T = Tsat
        if(SV.le.SVf) T = Tsat-0.00001
    5   IR = IR+1
        call SRSORT(Pref,T,SVnew,H,S,ISAT,SVg,Hg,Sg )
        del = (SVnew-SV)/SV
        if(abs(del).le.ERR) go to 10
        call bisection(IR,T,SVnew,SV,25.d0,del)
        go to 5
      else        
        stop ' two phase region, use other subroutine'
      endif
      
   10 X = (H-Hf)/(Hg-Hf)
      if(I2.eq.1) steamf = T                                                    ! 1, Temperature                                [oF]
      if(I2.eq.2) steamf = H                                                    ! 2, enthalpy                                   [Btu/lbm]
      if(I2.eq.3) steamf = SV                                                   ! 3, specific volume                            [ft3/lbm]
      if(I2 .eq. 4) then                                                        ! 4, viscocity                                  [lbm/ft/hr]
        if(X .le. 0.) then
          steamf = VISL(Pref,T)*3600.
        elseif(X .ge. 1.) then
          steamf = VISV(Pref,T)*3600.
        else
          steamf = VISL(Pref,T)*3600.
        endif
      endif
      if(I2 .eq. 5) then                                                        ! 5, thermal conductivity                       [Btu/ft/hr/oF]
        if(X .le. 0.) then
          steamf = CONDL(Pref,T)
        elseif(X .ge. 1.) then
          steamf = CONDV(Pref,T)
        endif
      endif
      if(I2 .eq. 6) then                                                        ! 6, specific heat                              [Btu/lbm/oF]
        if(X .le. 0.) then
          if(T .le. 662.) then
            steamf = CPPT1(Pref,T)
          else
            steamf = CPVT3(SV,T)
          endif
        elseif(X .ge. 1.) then
          if(T .le. 662.) then
            steamf = CPPT2(Pref,T)
          else
            PT = P23T(T)
            if(Pref .le. PT) then
              steamf = CPPT2(Pref,T)
            else
              steamf = CPVT3(SV,T)
            endif
          endif
        endif
      endif
      if(I2.eq.7) steamf = S                                                    ! 7, entropy                                    [Btu/lbm/oF]
      
      return                                               
      END                                                  

      subroutine bisection(IR,X,Y,YT,Xmin,del)                                  ! Pref  : Pressure                              [psia]
                                                                                ! I1    : given property option (1 or 2)
      implicit real*8 (a-h,o-z)                                                 ! I2    : wanted property option (1 ~ 7)
      
      save x1, x2

      if(IR .eq. 1) then
        X1 = 0.d0
        X2 = 0.d0
      endif
        
      if(X1.eq.0. .or. X2.eq.0.) then
        if(Y .lt. YT) then
          X1 = X
        else
          X2 = X
        endif
        n = 0
        if(del.gt.1.) n = log10(del)
	  X = X-del*X/10**n
        if(X .lt. Xmin) X = Xmin
        return
      else
        if(Y .lt. YT) then
          X1 = X
        else
          X2 = X
        endif
        if(X1 .gt. X2) stop ' X1 > X2 in bisection'
        X = (X1+X2)/2.
        return
      endif
      
      return
      end