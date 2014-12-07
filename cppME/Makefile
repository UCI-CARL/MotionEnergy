# MotionEnergy preliminary Makefile
#
# Ver 12/6/2014

.PHONY: all clean distclean
default: all

include libME.mk

motion_energy.o: include/motion_energy.h include/cuda_version_control.h
	nvcc -I/usr/local/cuda/samples/common/inc -Iinclude -D__CUDA6__ -arch sm_30 -use_fast_math src/motion_energy.cu -c motion_energy.o

# make all
all: motion_energy.o

clean:
	rm -rf motion_energy.o

distclean:
	rm -rf motion_energy.o tmp* *.log *.dat *.a results

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