# simple preliminary projects makefile
# uses ME library
# Ver 12/6/2014

default: main

main: src/main.cpp
	nvcc -I/opt/CARL/MotionEnergy/include -L/opt/CARL/MotionEnergy/lib $< -o $@ -lcuda -lcudart -lME

clean:
	rm -rf main

distclean:
	rm -rf main results/*