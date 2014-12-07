# MotionEnergy preliminary Makefile
#
# Ver 12/6/2014

me_major_num := 1
me_minor_num := 0
me_build_num := 0

targets :=
objects :=
libraries :=
output_files := doc/html


# this blank default is required
default:

include cppME/me.mk
include cppME/libME.mk

.PHONY: all clean distclean
default: $(targets)

clean:
	rm -rf $(objects)

distclean:
	rm -rf $(objects) $(targets) $(libraries) $(output_files)

# Print a help message
help:
	@ echo 
	@ echo '\TODO MotionEnergy Makefile options:'
	@ echo 
	@ echo "make            Compiles the MotionEnergy code using the default compiler"
	@ echo "make install    Installs MotionEnergy library (may require root privileges)"
	@ echo "make clean      Cleans out all object files"
	@ echo "make distclean  Cleans out all objects files and output files"
	@ echo "make help       Brings up this message!"