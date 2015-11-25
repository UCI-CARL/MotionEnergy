##----------------------------------------------------------------------------##
##
##   MotionEnergy Makefile
##   ----------------------
##
##   Author:    Michael Beyeler <mbeyeler@uci.edu>
##
##   Institute: Cognitive Anteater Robotics Lab (CARL)
##              Department of Cognitive Sciences
##              University of California, Irvine
##              Irvine, CA, 92697-5100, USA
##
##   Version:   11/25/2015
##
##----------------------------------------------------------------------------##
	
# the following are filled in the include files and passed up
targets :=
objects :=
libraries :=
output_files := doc/html

.PHONY: default clean distclean release debug
default: release

include configure.mk     # import configuration settings
include cppME/me.mk      # import ME-related variables and rules
include cppME/libME.mk   # import libME-related variables and rules

# clean all objects
clean:
	$(RM) $(objects)

# clean all objects and output files
distclean:
	$(RM) $(objects) $(targets) $(libraries) $(output_files)

# print a help message
help:
	@ echo 
	@ echo "MotionEnergy Makefile options:"
	@ echo 
	@ echo "make            Compiles ME in default mode (release)"
	@ echo "make release    Compiles ME in release mode (no debug output,"
	@ echo "                using fast math and GPU optimization level 3)"
	@ echo "make debug      Compiles ME in debug mode (-g -Wall)"
	@ echo "make install    Installs ME library (may require root privileges)"
	@ echo "make clean      Cleans out all object files"
	@ echo "make distclean  Cleans out all object and output files"
	@ echo "make help       Brings up this message"