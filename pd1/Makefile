all: bin

bin: main.cu ../util.h
	nvcc -o bin -I.. main.cu

clean:
	rm bin
