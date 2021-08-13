NVCC=nvcc

CFLAGS = -std=c++14 -g -I -O3

all: CNN.o layers.o networks.o
	$(NVCC) $(CFLAGS) -o cnn  CNN.o layers.o networks.o

CNN.o: CNN.cpp layers.hpp networks.hpp secure_float.hpp
	$(NVCC) $(CFLAGS) -c CNN.cpp

layers.o: layers.hpp secure_float.hpp

networks.o: networks.hpp layers.hpp

clean:
	rm *.o