# simple preliminary projects makefile
# uses ME library
# Ver 12/6/2014

default: main

main: src/main.cpp
	nvcc -I/opt/CARL/ME/include -L/opt/CARL/ME/lib -lME $< -o $@

clean:
	rm -rf main

distclean:
	rm -rf main results/*