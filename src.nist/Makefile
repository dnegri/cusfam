# Makefile for nist

PROGRAM  = libnist.a

SRCS     = aideal.f \
           ancond.f \
           anistf.f \
           aresid.f \
           bkcond.f \
           bndchk.f \
           dens.f \
           dens0.f \
           dfind.f \
           dfind0.f \
           dfind1.f \
           dfind2.f \
           dielec.f \
           dlsat.f \
           dlsatc.f \
           dvsat.f \
           dvsatc.f \
           hssolv.f \
           ivzero.f \
           kfact.f \
           pcoex.f \
           pdp.f \
           phsolv.f \
           pmelt.f \
           pmelt1.f \
           pmelt2.f \
           props.f \
           props0.f \
           props1.f \
           props2.f \
           psat.f \
           pssolv.f \
           psub.f \
           pvap.f \
           pvapc.f \
           rind.f \
           rind2.f \
           satnist.f \
           surf.f \
           tcoex.f \
           tmelt.f \
           tsat.f \
           tssolv.f \
           tsub.f \
           tvap.f \
           tvapc.f \
           visc.f \
           wtrans.f

SYSTEM   = "$(shell uname)"

FTN      = gfortran
FFFLAGS  = -O2 -fdefault-real-8
FFLAGS   = -DPAR_UNIX \
          $(FFFLAGS) -c

VPATH    = .
DEST     = .
INSTALL  = /usr/bin/install
LD       = ld
LDFLAGS  = -r
LIBS     = 
MAKEFILE = Makefile
PRINT    = pr
SHELL    = /bin/sh

OBJS     = $(SRCS:.f=.o)

.SUFFIXES: .f .f90 .o

.f.o:
		$(FTN) $(FFLAGS) $< -o ./$@

all:        compile
		@echo "Linking $(PROGRAM) ..."
		@$(LD) $(LDFLAGS) $(OBJS) $(LIBS) -o $(PROGRAM)
		@echo "done"

compile:     $(OBJS) $(LIBS)

clean:;     @rm -f $(OBJS) core *.a
