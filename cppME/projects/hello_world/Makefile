#------------------------------------------------------------------------------
# ME Project Makefile
#
# Note: This file depends on variables set in configure.mk, thus must be run
# after importing those others.
#------------------------------------------------------------------------------

# simple projects makefile
# Ver 11/25/2015

target := hello_world

# include all relevant variables, such as $NVCCINCFLAGS
include ../../configure.mk

.PHONY: default
default: $(target)

$(target): src/main_$(target).cpp
	$(NVCC) $(NVCCINCFLAGS) $(LIBINCFLAGS) $(NVCCFLAGS) $< -o $@ $(LIBLDFLAGS) $(NVCCLDFLAGS)

clean:
	$(RM) $(target)

distclean:
	$(RM) $(target) *.log *.dat results/*